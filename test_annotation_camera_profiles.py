"""Top cameras retain learned hand filters without another store's background."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

SCRIPTS = Path(__file__).parent / "auto-labeling" / "sam3_with_dino"
sys.path.insert(0, str(SCRIPTS))
try:
    from annotation_pipeline.gates import Gates
finally:
    sys.path.pop(0)


class CameraProfileTests(unittest.TestCase):
    def gate(self, reference=False):
        gate = object.__new__(Gates)
        gate.use_reference_background = reference
        gate.references = {"cam_0": Image.new("RGB", (640, 480))} if reference else {}
        gate.foreground = {}
        gate.sticker_score = Mock(side_effect=AssertionError("No calibrated background for top views"))
        return gate

    def test_cam1_retains_dino_hand_verification_and_patch_filter(self):
        gate = self.gate()
        hand = [[10, 20, 50, 70]]
        gate.dino = SimpleNamespace(verify=Mock(return_value={"hand": hand, "reason": "verified"}))
        gate.patch_keep = Mock(return_value={"keep": True})
        result = gate.run(Image.new("RGB", (640, 480)), "dino_fallback", hand, [], "cam_1")
        gate.dino.verify.assert_called_once()
        gate.patch_keep.assert_called_once()
        self.assertEqual(result["hand"], hand)
        self.assertIn("E006", result["decisions"])
        self.assertIn("E011", result["decisions"])

    def test_top_views_preserve_products_without_using_legacy_backgrounds(self):
        product = [[100, 110, 200, 240]]
        for cam in ["cam_0", "cam_1"]:
            gate = self.gate()
            result = gate.run(Image.new("RGB", (640, 480)), "none", [], product, cam)
            self.assertEqual(result["product"], product)
            self.assertFalse(result["decisions"]["E015"]["reject"])
            self.assertEqual(result["decisions"]["E015"]["reason"], "top_view_without_calibrated_reference")
            gate.sticker_score.assert_not_called()

    def test_reference_profile_still_requires_the_correct_camera(self):
        gate = self.gate(reference=True)
        with self.assertRaisesRegex(ValueError, "Unsupported ANNOTATION"):
            gate.run(Image.new("RGB", (640, 480)), "none", [], [], "cam_1")

    def test_top_profile_preserves_image_size_validation(self):
        gate = self.gate()
        with self.assertRaisesRegex(ValueError, "Unsupported ANNOTATION"):
            gate.run(Image.new("RGB", (1280, 960)), "none", [], [], "cam_1")


if __name__ == "__main__":
    unittest.main()
