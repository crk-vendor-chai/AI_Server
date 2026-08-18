"""동일 division 재요청이 자동 어노테이션을 안전하게 누적 처리하는지 검증.

시나리오는 실제 /v1/events/product/created 진입점을 그대로 태우고,
Mongo·GPU·MinIO·annotation job만 대역으로 바꾼다.
"""
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import main
import pipeline_run
from request_log import bind_request_context


class FakeBackgroundTasks:
    """FastAPI와 동일하게 응답 이후 등록 순서대로 task를 실행한다."""

    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, func, *args, **kwargs) -> None:
        self.tasks.append((func, args, kwargs))

    def run_pending(self) -> None:
        while self.tasks:
            func, args, kwargs = self.tasks.pop(0)
            func(*args, **kwargs)


class FakeGpuPool:
    total_slot_capacity = 4
    max_slots_per_gpu = 2
    slot_memory_fraction = 0.5

    def acquire(self, foldername: str) -> int:
        return 0

    def release(self, gpu_id: int, foldername: str) -> None:
        return None


def header(if_sysid: str) -> dict:
    return {
        "IF_ID": "IF_EDGE_01",
        "IF_SYSID": if_sysid,
        "IF_HOST": "edge",
        "IF_DATE": "20260803120000",
    }


class DivisionRerequestTests(unittest.TestCase):
    DIVISION = "DIVISION-1"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["PNT_PIPELINE_STATE_DIR"] = os.path.join(self.temp_dir.name, "state")
        os.environ["PNT_REQUEST_LOG_DIR"] = os.path.join(self.temp_dir.name, "logs")
        with pipeline_run._store_lock:
            pipeline_run._runs.clear()
            pipeline_run._loaded = True
        main.annotation_batch_queue.clear()
        main._annotation_active_run.clear()
        main._annotation_locks.clear()

        # division의 Mongo 상태: product_object_id -> training_status
        self.mongo_status = {}
        self.is_new = {}  # product_object_id -> isNew (기본 "1")
        self.annotated = []  # run_annotation_job 호출 순서
        self.has_artifact = True  # MinIO 검수 산출물 존재 여부

        self.stack = ExitStack()
        self.stack.enter_context(
            patch.object(main, "_resolve_product_info_from_mongo", self._fake_products)
        )
        self.stack.enter_context(
            patch.object(main, "run_annotation_job", self._fake_annotation_job)
        )
        self.stack.enter_context(
            patch.object(main, "get_gpu_slot_pool", lambda *a, **k: FakeGpuPool())
        )
        self.stack.enter_context(patch.object(main, "_has_product_seg_pkl", lambda p: False))
        self.stack.enter_context(
            patch.object(main, "_update_training_status_in_mongo", lambda *a, **k: True)
        )
        self.stack.enter_context(patch.object(main, "_maybe_advance_pipeline", lambda *a, **k: None))
        self.stack.enter_context(
            patch.object(
                pipeline_run,
                "has_new_annotation_artifact",
                lambda product: self.has_artifact,
            )
        )

    def tearDown(self) -> None:
        self.stack.close()
        # 삭제될 temp 로그 경로가 contextvar에 남으면 다음 테스트의 pipeline_log가 깨진다.
        bind_request_context(None)
        main.annotation_batch_queue.clear()
        main._annotation_active_run.clear()
        main._annotation_locks.clear()
        with pipeline_run._store_lock:
            pipeline_run._runs.clear()
            pipeline_run._loaded = False
        os.environ.pop("PNT_PIPELINE_STATE_DIR", None)
        os.environ.pop("PNT_REQUEST_LOG_DIR", None)
        self.temp_dir.cleanup()

    # --- 대역 -------------------------------------------------------------
    def _fake_products(self, division_idx, is_cold=False, raise_on_error=False):
        return [
            {
                "product_object_id": obj_id,
                "product_idx": f"P{obj_id[-1]}",
                "foldername": f"{obj_id[-1]}_PRODUCT",
                "train_product_idx": int(obj_id[-1]),
                "is_new": self.is_new.get(obj_id, "1"),
                "training_status": status,
                "brunch_name": f"{division_idx}_C",
            }
            for obj_id, status in sorted(self.mongo_status.items())
        ]

    def _fake_annotation_job(self, division_idx, product_info, **kwargs):
        self.annotated.append(product_info["product_object_id"])
        return True

    # --- 도우미 -----------------------------------------------------------
    def set_products(self, numbers, status="2"):
        for n in numbers:
            self.mongo_status[f"object-{n}"] = status

    def post_created(self, if_sysid: str) -> FakeBackgroundTasks:
        background = FakeBackgroundTasks()
        payload = main.ProductCreatedEnvelope(
            HEADER=header(if_sysid),
            DATA={"division_idx": self.DIVISION, "is_cold": True},
        )
        main.product_created(payload, background)
        return background

    def active_phases(self):
        run = pipeline_run.find_active_run(None, self.DIVISION, True)
        return {p["product_object_id"]: p["phase"] for p in run["products"]}

    # --- 시나리오 ---------------------------------------------------------
    def test_incremental_request_annotates_only_new_products(self) -> None:
        """1~4 어노테이션 완료(검수 대기) 후 5~7 추가 → 5~7만 어노테이션."""
        self.set_products([1, 2, 3, 4])
        first = self.post_created("REQUEST-1")
        first.run_pending()
        self.assertEqual(sorted(self.annotated), ["object-%d" % n for n in range(1, 5)])

        self.annotated.clear()
        self.set_products([5, 6, 7])
        second = self.post_created("REQUEST-2")
        second.run_pending()

        self.assertEqual(sorted(self.annotated), ["object-%d" % n for n in range(5, 8)])
        phases = self.active_phases()
        self.assertEqual(len(phases), 7)
        for n in range(1, 8):
            self.assertEqual(
                phases[f"object-{n}"],
                pipeline_run.PHASE_AWAITING_REVIEW,
                f"object-{n} 미완료: {phases}",
            )

    def test_full_list_rerequest_does_not_reannotate_completed_products(self) -> None:
        """1~4 완료 후 1~7 전체 재요청 → 5~7만 어노테이션, 1~4는 검수 대기 유지."""
        self.set_products([1, 2, 3, 4])
        self.post_created("REQUEST-1").run_pending()
        self.annotated.clear()

        self.set_products([5, 6, 7])  # Mongo에는 1~7이 모두 status=2로 남아 있음
        self.post_created("REQUEST-2").run_pending()

        self.assertEqual(sorted(self.annotated), ["object-%d" % n for n in range(5, 8)])
        run = pipeline_run.find_active_run(None, self.DIVISION, True)
        self.assertEqual(run["run_id"], "REQUEST-1")
        self.assertEqual(len(run["products"]), 7)

    def test_rerequest_while_first_batch_still_running(self) -> None:
        """배치1 실행 중 재요청이 들어와도 유실·중복 없이 이어서 처리."""
        self.set_products([1, 2])
        first = self.post_created("REQUEST-1")

        # 배치1이 도는 도중 5~7 추가 요청이 들어오는 상황을 재현한다.
        second_holder = {}

        def annotate_and_inject(division_idx, product_info, **kwargs):
            self.annotated.append(product_info["product_object_id"])
            if product_info["product_object_id"] == "object-1" and "bg" not in second_holder:
                self.set_products([5, 6])
                second_holder["bg"] = self.post_created("REQUEST-2")
            return True

        with patch.object(main, "run_annotation_job", annotate_and_inject):
            first.run_pending()
            second_holder["bg"].run_pending()

        self.assertEqual(sorted(self.annotated), ["object-1", "object-2", "object-5", "object-6"])
        self.assertEqual(len(self.active_phases()), 4)

    def test_failed_product_is_retried_by_next_request(self) -> None:
        """1차에서 실패한 상품은 재요청 배치에서 다시 어노테이션된다."""
        self.set_products([1, 2])

        def fail_object_2(division_idx, product_info, **kwargs):
            self.annotated.append(product_info["product_object_id"])
            return product_info["product_object_id"] != "object-2"

        with patch.object(main, "run_annotation_job", fail_object_2):
            self.post_created("REQUEST-1").run_pending()
        self.assertEqual(
            self.active_phases()["object-2"], pipeline_run.PHASE_FAILED
        )

        self.annotated.clear()
        self.set_products([3])
        self.post_created("REQUEST-2").run_pending()

        self.assertEqual(sorted(self.annotated), ["object-2", "object-3"])
        phases = self.active_phases()
        self.assertEqual(phases["object-2"], pipeline_run.PHASE_AWAITING_REVIEW)
        self.assertEqual(phases["object-3"], pipeline_run.PHASE_AWAITING_REVIEW)

    def test_missing_review_artifact_triggers_reannotation(self) -> None:
        """검수 산출물이 없으면 재요청 시 해당 상품을 다시 어노테이션한다."""
        self.set_products([1])
        self.post_created("REQUEST-1").run_pending()
        self.annotated.clear()

        self.has_artifact = False
        self.set_products([2])
        self.post_created("REQUEST-2").run_pending()

        self.assertEqual(sorted(self.annotated), ["object-1", "object-2"])

    def test_preceding_task_failure_does_not_block_annotation(self) -> None:
        """선행 background task가 죽어도 어노테이션 drain worker는 실행된다.

        FastAPI는 background task 중 하나가 예외를 내면 남은 task를 실행하지 않는다.
        product_created는 skip 파이프라인을 drain worker보다 먼저 등록하므로,
        격리가 없으면 skip 파이프라인 실패가 division 큐를 영구히 멈춘다.
        """
        self.set_products([1, 2])
        self.set_products([9])
        self.is_new["object-9"] = "0"  # isNew=0 → skip 파이프라인 대상

        def explode(*args, **kwargs):
            raise RuntimeError("mongo down")

        first = self.post_created("REQUEST-1")
        with patch.object(main, "_update_training_status_in_mongo", explode):
            first.run_pending()  # 예외가 요청 밖으로 새어나가지 않아야 한다

        self.assertEqual(sorted(self.annotated), ["object-1", "object-2"])

        self.annotated.clear()
        self.set_products([3])
        self.post_created("REQUEST-2").run_pending()

        self.assertEqual(
            sorted(self.annotated),
            ["object-3"],
            "재요청이 정체된 division 큐를 복구하지 못했습니다",
        )

    def test_dead_worker_releases_queue_for_next_request(self) -> None:
        """워커가 배치 도중 비정상 종료해도 남은 배치는 다음 요청이 이어서 처리한다."""
        self.set_products([1, 2])
        first = self.post_created("REQUEST-1")

        # 배치1 처리 중 재요청이 들어와 배치2가 같은 큐에 쌓인 뒤, 워커가 죽는 상황.
        def annotate_then_die(division_idx, product_info, **kwargs):
            self.annotated.append(product_info["product_object_id"])
            if "injected" not in self.is_new:
                self.is_new["injected"] = "1"
                self.set_products([5])
                self.post_created("REQUEST-2")
                raise KeyboardInterrupt("worker killed")
            return True

        with patch.object(main, "run_annotation_job", annotate_then_die):
            with self.assertRaises(KeyboardInterrupt):
                first.run_pending()

        queue_key = main._annotation_lock_key(self.DIVISION, True)
        self.assertEqual(main.annotation_batch_queue.pending_count(queue_key), 1)

        self.annotated.clear()
        self.set_products([6])
        self.post_created("REQUEST-3").run_pending()

        self.assertIn("object-5", self.annotated)
        self.assertIn("object-6", self.annotated)


if __name__ == "__main__":
    unittest.main()
