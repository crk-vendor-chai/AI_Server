import os
import subprocess
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import main
import segment_job
from gpu_scheduler import GpuSlotTimeout, NvmlGpuSlotPool


def snapshot(gpu_id):
    return {
        "gpu_id": gpu_id, "name": "test", "total_mib": 65536,
        "free_mib": 65536, "used_mib": 0, "utilization_pct": 0,
        "process_count": 0,
    }


def pool(**kwargs):
    return NvmlGpuSlotPool(
        gpu_ids=[0, 1], wait_timeout_sec=2, min_free_memory_mib=24576,
        max_utilization_pct=80, poll_interval_sec=0.01,
        status_log_interval_sec=60, snapshot_reader=snapshot, **kwargs,
    )


class SharedSlotTests(unittest.TestCase):
    def test_eight_mixed_jobs_share_four_slots(self):
        scheduler = pool()
        gate = threading.Event()
        full = threading.Event()
        lock = threading.Lock()
        active = Counter()
        maximum = Counter()

        def work(number):
            label = f"{'segment' if number % 2 else 'annotation'}:{number}"
            gpu_id = scheduler.acquire(label)
            try:
                with lock:
                    active[gpu_id] += 1
                    maximum[gpu_id] = max(maximum[gpu_id], active[gpu_id])
                    if sum(active.values()) == 4:
                        full.set()
                if not gate.wait(2):
                    raise RuntimeError("test gate timed out")
            finally:
                with lock:
                    active[gpu_id] -= 1
                scheduler.release(gpu_id, label)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(work, n) for n in range(8)]
            try:
                self.assertTrue(full.wait(1), "first four jobs did not acquire slots")
                with lock:
                    self.assertEqual(dict(active), {0: 2, 1: 2})
                self.assertEqual(scheduler._reserved, {0: 2, 1: 2})
            finally:
                gate.set()
            for future in futures:
                future.result(timeout=3)
        self.assertEqual(maximum, {0: 2, 1: 2})
        self.assertEqual(scheduler._reserved, {0: 0, 1: 0})

    def test_memory_pressure_selects_other_gpu(self):
        scheduler = pool()

        def busy_first(gpu_id):
            state = snapshot(gpu_id)
            if gpu_id == 0:
                state["free_mib"] = 728
            return state

        scheduler._snapshot_reader = busy_first
        gpu_id = scheduler.acquire("segment:63")
        self.assertEqual(gpu_id, 1)
        scheduler.release(gpu_id)


class SegmentOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = pool()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.get_pool = self.stack.enter_context(
            patch.object(main, "get_gpu_slot_pool", return_value=self.scheduler)
        )
        for name in ("_bind_request_ctx", "pipeline_log", "find_active_run", "_maybe_advance_pipeline"):
            self.stack.enter_context(patch.object(main, name))
        self.phase = self.stack.enter_context(patch.object(main, "update_product_phase"))
        self.status = self.stack.enter_context(
            patch.object(main, "_update_training_status_in_mongo", return_value=True)
        )
        self.job = self.stack.enter_context(patch.object(main, "run_segment_job", return_value=True))
        self.release = self.stack.enter_context(
            patch.object(self.scheduler, "release", wraps=self.scheduler.release)
        )
        self.product = {"product_object_id": "object-63", "product_idx": "P63", "foldername": "63_PRODUCT"}

    def run_job(self):
        main._run_segment_job("DIVISION", self.product, True, run_id="RUN")

    def test_success_reserves_gpu_and_releases_before_advance(self):
        def segment(*args, **kwargs):
            self.assertEqual(self.scheduler._reserved[kwargs["gpu_id"]], 1)
            return True

        self.job.side_effect = segment
        self.run_job()
        self.get_pool.assert_called_once_with(
            main._annotation_gpu_ids(), main._annotation_gpu_slot_wait_sec(), logger=main.pipeline_log
        )
        self.assertEqual(self.job.call_args.kwargs["gpu_id"], 0)
        self.release.assert_called_once_with(0, "segment:63_PRODUCT")
        self.assertEqual(self.scheduler._reserved, {0: 0, 1: 0})
        self.status.assert_called_once_with("DIVISION_C", "object-63", status="4")
        self.assertEqual(self.phase.call_args.args[-1], main.PHASE_READY)

    def test_job_failure_and_exception_release_slot_and_mark_failed(self):
        for failure in (False, RuntimeError("CUDA out of memory")):
            with self.subTest(failure=failure):
                self.job.side_effect = failure if isinstance(failure, Exception) else None
                self.job.return_value = False
                self.run_job()
                self.assertEqual(self.scheduler._reserved, {0: 0, 1: 0})
                self.status.assert_not_called()
                self.assertEqual(self.phase.call_args.args[-1], main.PHASE_FAILED)
        self.assertEqual(self.release.call_count, 2)

    def test_slot_timeout_does_not_launch_or_release_unowned_slot(self):
        with patch.object(self.scheduler, "acquire", side_effect=GpuSlotTimeout("busy")):
            self.run_job()
        self.job.assert_not_called()
        self.release.assert_not_called()
        self.assertEqual(self.phase.call_args.args[-1], main.PHASE_FAILED)

    def test_status_update_failure_does_not_report_ready(self):
        self.status.return_value = False
        self.run_job()
        self.assertEqual(self.scheduler._reserved, {0: 0, 1: 0})
        self.assertEqual(self.phase.call_args.args[-1], main.PHASE_FAILED)


class SegmentDeviceTests(unittest.TestCase):
    def test_selected_device_overrides_inherited_device(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            script = Path(temp) / "segment.sh"
            script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            stack.enter_context(patch.dict(os.environ, {
                "SEGMENT_BASE_DIR": temp, "ANNOTATION_BASE_DIR": temp,
                "SEGMENT_SCRIPT": str(script), "PIPELINE_CUDA_VISIBLE_DEVICES": "1",
                "CUDA_VISIBLE_DEVICES": "1",
            }))
            stack.enter_context(patch.object(segment_job, "_count_local_images", return_value=1))
            stack.enter_context(patch.object(segment_job, "download_labels_from_minio", return_value=1))
            stack.enter_context(patch.object(segment_job, "get_request_context", return_value=None))
            shell = stack.enter_context(patch.object(
                segment_job, "run_shell_job", return_value=subprocess.CompletedProcess([], 0)
            ))
            self.assertTrue(segment_job.run_segment_job("DIVISION", {
                "foldername": "63_PRODUCT", "train_product_idx": 63,
                "product_eng_name": "PRODUCT",
            }, gpu_id=0))
            self.assertEqual(shell.call_args.kwargs["env"]["PIPELINE_CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(shell.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
