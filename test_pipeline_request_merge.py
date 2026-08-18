import os
import tempfile
import unittest
from unittest.mock import patch

import pipeline_run
from annotation_batch_queue import SerialBatchQueue


def product(number: int) -> dict:
    return {
        "product_object_id": f"object-{number}",
        "product_idx": f"P{number}",
        "foldername": f"{number}_PRODUCT",
        "train_product_idx": number,
        "is_new": "1",
    }


class PipelineRequestMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["PNT_PIPELINE_STATE_DIR"] = self.temp_dir.name
        with pipeline_run._store_lock:
            pipeline_run._runs.clear()
            pipeline_run._loaded = True

    def tearDown(self) -> None:
        with pipeline_run._store_lock:
            pipeline_run._runs.clear()
            pipeline_run._loaded = False
        os.environ.pop("PNT_PIPELINE_STATE_DIR", None)
        self.temp_dir.cleanup()

    def register(self, run_id: str, division_idx: str, products: list[dict]):
        return pipeline_run.merge_or_create_run_from_created(
            None,
            run_id=run_id,
            if_id="IF_EDGE_01",
            division_idx=division_idx,
            is_cold=True,
            brunch_name=f"{division_idx}_C",
            log_dir=f"/tmp/{run_id}",
            pending_annotation=products,
            pending_skip=[],
            pending_segment=[],
        )

    def test_repeated_request_merges_products_without_regressing_progress(self) -> None:
        run, effective_run_id, merged = self.register(
            "REQUEST-1", "DIVISION-1", [product(i) for i in range(1, 5)]
        )
        self.assertFalse(merged)
        self.assertEqual(effective_run_id, "REQUEST-1")
        self.assertEqual(len(run["products"]), 4)

        pipeline_run.update_product_phase(
            None, "REQUEST-1", "object-1", pipeline_run.PHASE_AWAITING_REVIEW
        )
        pipeline_run.update_product_phase(
            None, "REQUEST-1", "object-2", pipeline_run.PHASE_FAILED
        )
        pipeline_run.update_product_phase(
            None, "REQUEST-1", "object-3", pipeline_run.PHASE_SEGMENTING
        )

        with patch.object(pipeline_run, "has_new_annotation_artifact", return_value=True):
            run, effective_run_id, merged = self.register(
                "REQUEST-2", "DIVISION-1", [product(i) for i in range(1, 8)]
            )

        self.assertTrue(merged)
        self.assertEqual(effective_run_id, "REQUEST-1")
        self.assertEqual(len(run["products"]), 7)
        phases = {item["product_object_id"]: item["phase"] for item in run["products"]}
        self.assertEqual(phases["object-1"], pipeline_run.PHASE_AWAITING_REVIEW)
        self.assertEqual(phases["object-2"], pipeline_run.PHASE_PENDING_ANNOTATION)
        self.assertEqual(phases["object-3"], pipeline_run.PHASE_SEGMENTING)
        self.assertEqual(phases["object-4"], pipeline_run.PHASE_PENDING_ANNOTATION)
        self.assertEqual(phases["object-7"], pipeline_run.PHASE_PENDING_ANNOTATION)

        original = pipeline_run.find_run_by_id(None, "REQUEST-1")
        merged_request = pipeline_run.find_run_by_id(None, "REQUEST-2")
        self.assertEqual(original["run_status"], pipeline_run.RUN_IN_PROGRESS)
        self.assertEqual(merged_request["run_status"], pipeline_run.RUN_MERGED)
        self.assertEqual(merged_request["merged_into"], "REQUEST-1")

    def test_different_divisions_remain_independent(self) -> None:
        self.register("STORE-A", "DIVISION-A", [product(1)])
        self.register("STORE-B", "DIVISION-B", [product(1)])

        self.assertEqual(
            pipeline_run.find_active_run(None, "DIVISION-A", True)["run_id"], "STORE-A"
        )
        self.assertEqual(
            pipeline_run.find_active_run(None, "DIVISION-B", True)["run_id"], "STORE-B"
        )

    def test_missing_review_artifact_is_requeued(self) -> None:
        self.register("REQUEST-1", "DIVISION-1", [product(1)])
        pipeline_run.update_product_phase(
            None, "REQUEST-1", "object-1", pipeline_run.PHASE_AWAITING_REVIEW
        )

        with patch.object(pipeline_run, "has_new_annotation_artifact", return_value=False):
            run, _, _ = self.register("REQUEST-2", "DIVISION-1", [product(1)])

        self.assertEqual(
            pipeline_run.get_product_phase(run, "object-1"),
            pipeline_run.PHASE_PENDING_ANNOTATION,
        )

    def test_same_if_sysid_is_scoped_when_divisions_differ(self) -> None:
        _, first_run_id, _ = self.register("SAME-ID", "DIVISION-A", [product(1)])
        second, second_run_id, _ = self.register("SAME-ID", "DIVISION-B", [product(1)])

        self.assertEqual(first_run_id, "SAME-ID")
        self.assertNotEqual(second_run_id, first_run_id)
        self.assertEqual(second["source_run_id"], "SAME-ID")
        self.assertEqual(
            pipeline_run.find_active_run(None, "DIVISION-A", True)["run_id"], first_run_id
        )
        self.assertEqual(
            pipeline_run.find_active_run(None, "DIVISION-B", True)["run_id"], second_run_id
        )


class SerialBatchQueueTests(unittest.TestCase):
    def test_one_worker_drains_batches_in_order(self) -> None:
        queue = SerialBatchQueue()
        token, count = queue.enqueue("DIV-1:C", "batch-1")
        self.assertIsNotNone(token)
        self.assertEqual(count, 1)
        self.assertEqual(queue.enqueue("DIV-1:C", "batch-2"), (None, 2))
        other_token, other_count = queue.enqueue("DIV-2:C", "other")
        self.assertIsNotNone(other_token)
        self.assertEqual(other_count, 1)

        self.assertEqual(queue.pop("DIV-1:C", token), "batch-1")
        self.assertEqual(queue.pop("DIV-1:C", token), "batch-2")
        self.assertIsNone(queue.pop("DIV-1:C", token))

        next_token, next_count = queue.enqueue("DIV-1:C", "batch-3")
        self.assertIsNotNone(next_token)
        self.assertEqual(next_count, 1)
        self.assertNotEqual(next_token, token)

    def test_dead_worker_retires_and_leaves_pending_batches(self) -> None:
        queue = SerialBatchQueue()
        token, _ = queue.enqueue("DIV-1:C", "batch-1")
        queue.enqueue("DIV-1:C", "batch-2")
        self.assertEqual(queue.pop("DIV-1:C", token), "batch-1")

        # 워커가 batch-2를 남기고 죽은 상황
        self.assertTrue(queue.retire("DIV-1:C", token))
        self.assertEqual(queue.pending_count("DIV-1:C"), 1)

        new_token, count = queue.enqueue("DIV-1:C", "batch-3")
        self.assertIsNotNone(new_token)
        self.assertEqual(count, 2)
        self.assertEqual(queue.pop("DIV-1:C", new_token), "batch-2")
        self.assertEqual(queue.pop("DIV-1:C", new_token), "batch-3")

    def test_retire_does_not_steal_key_from_successor_worker(self) -> None:
        queue = SerialBatchQueue()
        token, _ = queue.enqueue("DIV-1:C", "batch-1")
        self.assertEqual(queue.pop("DIV-1:C", token), "batch-1")
        self.assertIsNone(queue.pop("DIV-1:C", token))  # 정상 은퇴

        successor, _ = queue.enqueue("DIV-1:C", "batch-2")
        self.assertFalse(queue.retire("DIV-1:C", token))  # 뒤늦은 finally
        self.assertIsNone(queue.enqueue("DIV-1:C", "batch-3")[0])  # 워커 중복 기동 없음
        self.assertEqual(queue.pop("DIV-1:C", successor), "batch-2")


if __name__ == "__main__":
    unittest.main()
