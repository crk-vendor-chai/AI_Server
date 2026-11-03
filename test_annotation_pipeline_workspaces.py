"""Exercise concurrent product runners without loading models or using MinIO."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


SCRIPTS = Path(__file__).parent / "auto-labeling" / "sam3_with_dino"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
try:
    runner = load_script("annotation_runner", "annotation_pipeline.py")
    asis = load_script("isolated_asis", "asis_pipeline.py")
finally:
    sys.path.pop(0)


class AnnotationWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.runs = self.root / "runs"
        self.products = ["1_PRODUCT", "2_PRODUCT"]
        for product in self.products:
            camera = self.images / product / "cam_0"
            camera.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(camera / "000001.jpg")
        self.barrier = None

    def fake_engine(self, command, log_path, env):
        def option(name):
            return command[command.index(name) + 1]

        product = option("--product")
        run_name = option("--run-name")
        if self.barrier and run_name.endswith("__pass1"):
            self.barrier.wait(timeout=10)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation makes unintended log sharing fail the test.
        with log_path.open("x") as log:
            log.write(product)
        camera = Path(option("--runs-root")) / run_name / product / "cam_0"
        for directory in ("debug", "labels", "labels_seg"):
            (camera / directory).mkdir(parents=True, exist_ok=True)
        (camera / "debug/detections.jsonl").write_text(
            json.dumps({"image": "000001.jpg", "best_hand": None}) + "\n"
        )
        (camera / "labels/000001.txt").write_text(
            product.split("_", 1)[0] + " 0.5 0.5 0.2 0.2\n"
        )
        cache = Path(option("--product-cache-root"))
        (cache / (product + ".npz")).write_text(product)
        if "--hand-seed-fill-root" in command:
            self.assertTrue((Path(option("--hand-seed-fill-root")) / (product.split("_")[0] + "_cam_0.json")).is_file())

    def arguments(self, product):
        return [
            "--img-root", str(self.images), "--runs-root", str(self.runs),
            "--run-name", "shared-run", "--product", product,
            "--cams", "cam_0", "--skip-export",
        ]

    def test_same_timestamp_reserves_distinct_directories_concurrently(self):
        legacy = self.runs / "shared-run__annotation"
        legacy.mkdir(parents=True)
        (legacy / "asis.log").write_text("previous execution")
        with patch.object(runner, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261026_100000_000001"
            with ThreadPoolExecutor(max_workers=8) as executor:
                paths = list(executor.map(lambda _: runner.create_work_dir(self.runs, "shared-run"), range(32)))
        self.assertEqual(len(set(paths)), 32)
        self.assertTrue(all(p.is_dir() and p.name.startswith("shared-run__annotation_20261026_100000_000001_") for p in paths))
        self.assertEqual((legacy / "asis.log").read_text(), "previous execution")

    def test_parallel_products_isolate_logs_caches_and_keep_result_paths(self):
        self.barrier = threading.Barrier(2)
        (self.root / "manifest.json").write_text("{}")
        bundle = SimpleNamespace(folder=self.root, manifest={"release": "test-release"})

        def fake_subprocess(command, log_path, env):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("x") as log:
                log.write(" ".join(command))
            if Path(command[1]).name == "asis_pipeline.py":
                self.assertEqual(asis.main(command[2:]), 0)

        with patch.object(runner, "Bundle", return_value=bundle), patch.object(runner, "run", side_effect=fake_subprocess), patch.object(asis, "_run", side_effect=self.fake_engine):
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(list(executor.map(lambda p: runner.main(self.arguments(p)), self.products)), [0, 0])

        workspaces = list(self.runs.glob("shared-run__annotation_*"))
        self.assertEqual(len(workspaces), 2)
        seen_products = set()
        for workspace in workspaces:
            summary = json.loads((workspace / "asis/asis_summary.json").read_text())
            product = next(iter(summary["postprocess"])).split("/")[0]
            seen_products.add(product)
            self.assertEqual((workspace / "asis/pass1.log").read_text(), product)
            self.assertEqual((workspace / "asis/pass2.log").read_text(), product)
            self.assertEqual([p.name for p in (workspace / "asis/cache/product").iterdir()], [product + ".npz"])
            self.assertTrue((self.runs / "shared-run" / product / "ANNOTATION_COMPLETE.json").is_file())
            self.assertTrue((self.runs / "shared-run__pass1" / product / "cam_0/debug/detections.jsonl").is_file())
        self.assertEqual(seen_products, set(self.products))
        self.assertFalse((self.runs / "shared-run/asis_pipeline.log").exists())
        self.assertFalse((self.runs / "shared-run/asis_summary.json").exists())
        self.assertFalse((self.runs / "_asis_cache").exists())

    def test_existing_product_output_is_still_protected(self):
        output = self.runs / "shared-run" / self.products[0]
        output.mkdir(parents=True)
        with patch.object(runner, "Bundle"), patch.object(runner, "run") as run:
            with self.assertRaisesRegex(FileExistsError, "Refuse to overwrite"):
                runner.main(self.arguments(self.products[0]))
        run.assert_not_called()

    def test_standalone_asis_keeps_legacy_paths(self):
        with patch.object(asis, "_run", side_effect=self.fake_engine):
            self.assertEqual(asis.main(self.arguments(self.products[0])), 0)
        self.assertTrue((self.runs / "shared-run__pass1/asis_pipeline.log").is_file())
        self.assertTrue((self.runs / "shared-run/asis_pipeline.log").is_file())
        self.assertTrue((self.runs / "shared-run/asis_summary.json").is_file())
        self.assertTrue((self.runs / "_asis_cache/shared-run/product").is_dir())


if __name__ == "__main__":
    unittest.main()
