from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "auto-labeling" / "sam3_with_dino" / "asis_pipeline.py"
SPEC = importlib.util.spec_from_file_location("asis_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
asis_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(asis_pipeline)


class AsisPipelineTests(unittest.TestCase):
    def test_default_engine_is_packaged_with_pipeline(self) -> None:
        engine = asis_pipeline._resolve_engine(None)
        self.assertEqual(engine.name, "run_product.py")
        self.assertEqual(engine.parent, MODULE_PATH.parent)

    def test_wrapper_options_are_not_forwarded_to_engine(self) -> None:
        argv = [
            "--product", "45_PRODUCT",
            "--run-name=api-run",
            "--asis-fill-threshold", "0.5",
            "--skip-export",
            "--gpu", "1",
        ]
        self.assertEqual(
            asis_pipeline._strip_wrapper_options(argv),
            ["--product", "45_PRODUCT", "--gpu", "1"],
        )

    def test_short_hand_runs_support_prefixed_frame_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            hand = "0 0.5 0.5 0.2 0.2\n"
            product = "45 0.5 0.5 0.2 0.2\n"
            for index in (1, 2):
                (labels / f"frame_{index:06d}.txt").write_text(hand + product)
            (labels / "frame_000003.txt").write_text(product)
            for index in (4, 5, 6, 7):
                (labels / f"frame_{index:06d}.txt").write_text(hand + product)

            removed = asis_pipeline._filter_short_hand_runs(labels, maximum=3)

            self.assertEqual(removed, 2)
            self.assertNotIn("0 ", (labels / "frame_000001.txt").read_text())
            self.assertIn("0 ", (labels / "frame_000004.txt").read_text())


if __name__ == "__main__":
    unittest.main()
