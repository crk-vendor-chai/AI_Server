import unittest
from unittest.mock import patch

import main


class FakeBackgroundTasks:
    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, func, *args, **kwargs) -> None:
        self.tasks.append((func, args, kwargs))


class AnnotationRequestQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        main.annotation_batch_queue.clear()

    def tearDown(self) -> None:
        main.annotation_batch_queue.clear()

    def test_repeated_requests_share_one_worker_and_keep_batch_order(self) -> None:
        first = {"product_object_id": "object-1", "foldername": "1_PRODUCT"}
        second = {"product_object_id": "object-2", "foldername": "2_PRODUCT"}
        background = FakeBackgroundTasks()

        main._schedule_annotation_batch(
            background, "DIVISION-1", [first, first], True, None, "RUN-1"
        )
        main._schedule_annotation_batch(
            background, "DIVISION-1", [first, second], True, None, "RUN-1"
        )

        self.assertEqual(len(background.tasks), 1)
        drain, args, kwargs = background.tasks[0]
        with patch.object(main, "_run_annotation_after_created") as run_batch:
            drain(*args, **kwargs)

        self.assertEqual(run_batch.call_count, 2)
        first_products = run_batch.call_args_list[0].args[1]
        second_products = run_batch.call_args_list[1].args[1]
        self.assertEqual([p["product_object_id"] for p in first_products], ["object-1"])
        self.assertEqual(
            [p["product_object_id"] for p in second_products],
            ["object-1", "object-2"],
        )

    def test_different_divisions_start_independent_workers(self) -> None:
        product = {"product_object_id": "same-product", "foldername": "1_PRODUCT"}
        background = FakeBackgroundTasks()

        main._schedule_annotation_batch(
            background, "DIVISION-A", [product], True, None, "RUN-A"
        )
        main._schedule_annotation_batch(
            background, "DIVISION-B", [product], True, None, "RUN-B"
        )

        self.assertEqual(len(background.tasks), 2)


if __name__ == "__main__":
    unittest.main()
