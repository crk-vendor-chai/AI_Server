from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_job import _build_review_image_manifest, _normalize_cam_upload_folder
from export_web_annotations import build_payload, normalize_cam_upload_folder


class CameraPathTests(unittest.TestCase):
    def test_camera_names_keep_source_underscores(self):
        self.assertEqual(_normalize_cam_upload_folder("cam_0"), "cam_0")
        self.assertEqual(_normalize_cam_upload_folder("cam_2"), "cam_2")
        self.assertEqual(normalize_cam_upload_folder("cam_0"), "cam_0")
        self.assertEqual(normalize_cam_upload_folder("cam_2"), "cam_2")

    def test_coco_and_review_manifest_use_cam_underscore_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            product_name = "47_BOTTLE_TEST_20260803_120000"
            image_root = root / "images" / product_name
            results_root = root / "results"
            root_folder = "DI_TEST_C_47_P_TEST"

            for camera in ("cam_0", "cam_2"):
                source_dir = image_root / camera
                label_dir = results_root / camera / "labels"
                source_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                Image.new("RGB", (16, 12), color=(30, 60, 90)).save(
                    source_dir / "000001.jpg"
                )
                (label_dir / "000001.txt").write_text(
                    "0 0.500000 0.500000 0.250000 0.250000\n",
                    encoding="utf-8",
                )

            payload = build_payload(
                SimpleNamespace(
                    results_root=str(results_root),
                    image_root=str(image_root),
                    product_class_id=47,
                    product_name="BOTTLE_TEST",
                    root_folder=root_folder,
                    brunch_name="",
                    train_product_idx=47,
                    product_idx="",
                    cams=["cam_0", "cam_2"],
                )
            )
            file_names = {image["file_name"] for image in payload["images"]}
            self.assertEqual(
                file_names,
                {
                    f"{root_folder}/cam_0/000001.jpg",
                    f"{root_folder}/cam_2/000001.jpg",
                },
            )
            self.assertFalse(any("/cam0/" in name or "/cam2/" in name for name in file_names))

            annotations_path = root / "annotations_bottle_test.json"
            annotations_path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = _build_review_image_manifest(
                annotations_path,
                image_root,
                root_folder,
            )
            self.assertIsNotNone(manifest)
            self.assertEqual(
                {camera for camera, _ in manifest or []},
                {"cam_0", "cam_2"},
            )


if __name__ == "__main__":
    unittest.main()
