from pathlib import Path
import tempfile
import unittest

from class_schema import SYNTHETIC_CLASS_ID_OFFSET
from training_job import (
    _build_canonical_class_names,
    _build_source_class_maps,
    _remap_label_class_ids,
)


class TrainingClassSchemaTest(unittest.TestCase):
    def setUp(self):
        self.deployed_names = [
            "hand",
            "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
            "CUP_BIBIGO_TTEOKBOKKI_110G",
            "BAG_GAMBAS_RICE_230G",
            "FROZEN_BAG_GAMBAS_RICE_230G",
        ]
        self.dataset_key = (
            "syn_100013_BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G"
        )
        self.new_product_key = "syn_100014_BOX_BINGGRAE_YOMAMTE_150ML"
        shifted_schema = {
            0: "hand",
            1: "CUP_BIBIGO_TTEOKBOKKI_110G",
            2: "BAG_GAMBAS_RICE_230G",
            3: "FROZEN_BAG_GAMBAS_RICE_230G",
            4: "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_COLD_200G",
            5: "BOX_BINGGRAE_YOMAMTE_150ML",
        }
        self.review_schemas = {
            self.dataset_key: shifted_schema,
            self.new_product_key: shifted_schema,
        }
        self.products = [
            {
                "train_product_idx": 100013,
                "product_eng_name":
                    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
            },
            {
                "train_product_idx": 100014,
                "product_eng_name": "BOX_BINGGRAE_YOMAMTE_150ML",
            },
        ]

    def test_shifted_cvat_ids_map_by_name(self):
        names = _build_canonical_class_names(
            self.deployed_names,
            self.review_schemas,
        )
        review_maps, synthetic_map, product_map = _build_source_class_maps(
            self.products,
            names,
            self.review_schemas,
        )

        self.assertEqual(names[0], "hand")
        self.assertEqual(names[1], self.deployed_names[1])
        self.assertEqual(names[5], "BOX_BINGGRAE_YOMAMTE_150ML")
        self.assertEqual(review_maps[self.dataset_key][1], 2)
        self.assertEqual(review_maps[self.dataset_key][4], 1)
        self.assertEqual(synthetic_map, {100013: 1, 100014: 5})
        self.assertEqual(product_map, {1: 100013, 5: 100014})

        with tempfile.TemporaryDirectory() as tmp:
            labels_root = Path(tmp)
            label_dir = labels_root / self.dataset_key / "labels"
            label_dir.mkdir(parents=True)
            label_path = label_dir / "sample.txt"
            label_path.write_text(
                "4 0.5 0.5 0.2 0.2\n"
                f"{SYNTHETIC_CLASS_ID_OFFSET + 100014} 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )

            checked, dropped, dist = _remap_label_class_ids(
                labels_root,
                review_maps,
                synthetic_map,
            )

            self.assertEqual((checked, dropped), (1, 0))
            self.assertEqual(dist, {1: 1, 5: 1})
            self.assertEqual(
                [line.split()[0] for line in label_path.read_text().splitlines()],
                ["1", "5"],
            )


if __name__ == "__main__":
    unittest.main()
