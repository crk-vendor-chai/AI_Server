from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import synthesis_job


class FakeMinio:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def fput_object(self, bucket: str, object_name: str, local_path: str) -> None:
        self.uploads.append((bucket, object_name, local_path))


class SynthesisJobTests(unittest.TestCase):
    def test_upload_includes_synthesized_and_raw_fallback_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            for stem in ("syn00001", "raw00002"):
                (images / f"{stem}.jpg").write_bytes(b"image")
                (labels / f"{stem}.txt").write_text("")
            client = FakeMinio()

            ok = synthesis_job._upload_synthesis_to_minio(
                client, "bucket", "DIV", "45", "PRODUCT", images, labels
            )

            self.assertTrue(ok)
            names = {item[1].rsplit("/", 1)[-1] for item in client.uploads}
            self.assertEqual(
                names,
                {"syn00001.jpg", "syn00001.txt", "raw00002.jpg", "raw00002.txt"},
            )

    def test_upload_rejects_unpaired_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "syn00001.jpg").write_bytes(b"image")
            client = FakeMinio()

            ok = synthesis_job._upload_synthesis_to_minio(
                client, "bucket", "DIV", "45", "PRODUCT", images, labels
            )

            self.assertFalse(ok)
            self.assertEqual(client.uploads, [])

    def test_unknown_blur_profile_is_rejected(self) -> None:
        segment = np.zeros((16, 16, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            synthesis_job.apply_segment_shake_blur(segment, prob=1.0, profile="unknown")


if __name__ == "__main__":
    unittest.main()
