from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


GpuSnapshot = Dict[str, Any]
SnapshotReader = Callable[[int], GpuSnapshot]
LogWriter = Callable[..., None]


class GpuSlotTimeout(RuntimeError):
    pass


_nvml_init_lock = threading.Lock()
_nvml_module = None


def _get_nvml():
    """Initialize NVML once per API process."""
    global _nvml_module
    if _nvml_module is not None:
        return _nvml_module

    with _nvml_init_lock:
        if _nvml_module is not None:
            return _nvml_module
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception as exc:
            raise RuntimeError(
                f"NVML initialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        _nvml_module = pynvml
        return _nvml_module


def read_gpu_snapshot(gpu_id: int) -> GpuSnapshot:
    """Read current memory, utilization and process state for a physical GPU."""
    pynvml = _get_nvml()
    device_count = pynvml.nvmlDeviceGetCount()
    if gpu_id < 0 or gpu_id >= device_count:
        raise RuntimeError(
            f"GPU {gpu_id} does not exist (NVML device_count={device_count})"
        )

    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    process_ids = set()
    for reader_name in (
        "nvmlDeviceGetComputeRunningProcesses",
        "nvmlDeviceGetGraphicsRunningProcesses",
    ):
        reader = getattr(pynvml, reader_name, None)
        if reader is None:
            continue
        try:
            process_ids.update(process.pid for process in reader(handle))
        except pynvml.NVMLError:
            # Process visibility is informative; memory/utilization remain authoritative.
            continue

    mib = 1024 * 1024
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "gpu_id": gpu_id,
        "name": str(name),
        "total_mib": int(memory.total // mib),
        "used_mib": int(memory.used // mib),
        "free_mib": int(memory.free // mib),
        "utilization_pct": int(utilization.gpu),
        "process_count": len(process_ids),
    }


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


class NvmlGpuSlotPool:
    """Select GPUs using live NVML state and per-device fractional slots."""

    def __init__(
        self,
        gpu_ids: Sequence[int],
        wait_timeout_sec: int,
        min_free_memory_mib: int,
        max_utilization_pct: int,
        poll_interval_sec: float,
        status_log_interval_sec: float,
        snapshot_reader: Optional[SnapshotReader] = None,
        logger: Optional[LogWriter] = None,
        slot_memory_fraction: float = 0.5,
        max_slots_per_gpu: int = 2,
    ):
        if not gpu_ids:
            raise ValueError("At least one GPU id is required")

        self.gpu_ids: Tuple[int, ...] = tuple(gpu_ids)
        self.wait_timeout_sec = wait_timeout_sec
        self.min_free_memory_mib = min_free_memory_mib
        self.max_utilization_pct = max_utilization_pct
        self.poll_interval_sec = poll_interval_sec
        self.status_log_interval_sec = status_log_interval_sec
        self.slot_memory_fraction = slot_memory_fraction
        self.max_slots_per_gpu = max_slots_per_gpu
        self._snapshot_reader = snapshot_reader or read_gpu_snapshot
        self._logger = logger
        self._condition = threading.Condition()
        self._reserved: Dict[int, int] = {gpu_id: 0 for gpu_id in self.gpu_ids}

        snapshots = [self._snapshot_reader(gpu_id) for gpu_id in self.gpu_ids]
        devices = ", ".join(
            f"GPU{s['gpu_id']}={s['name']}({s['total_mib']}MiB)" for s in snapshots
        )
        self._log(
            f"NVML GPU scheduler initialized: {devices}; "
            f"slot_memory={self.slot_memory_fraction:.0%}, "
            f"max_slots_per_gpu={self.max_slots_per_gpu}, "
            f"min_free={self.min_free_memory_mib}MiB, "
            f"max_util={self.max_utilization_pct}%, "
            f"poll={self.poll_interval_sec:g}s"
        )

    @property
    def total_slot_capacity(self) -> int:
        return len(self.gpu_ids) * self.max_slots_per_gpu

    def _required_free_mib(self, snapshot: GpuSnapshot) -> int:
        fractional = math.ceil(
            float(snapshot["total_mib"]) * self.slot_memory_fraction
        )
        return max(self.min_free_memory_mib, fractional)

    @classmethod
    def from_env(
        cls,
        gpu_ids: Sequence[int],
        wait_timeout_sec: int,
        logger: Optional[LogWriter] = None,
    ) -> "NvmlGpuSlotPool":
        return cls(
            gpu_ids=gpu_ids,
            wait_timeout_sec=wait_timeout_sec,
            min_free_memory_mib=_env_int(
                "ANNOTATION_GPU_MIN_FREE_MEMORY_MIB", 24576, 0, 1048576
            ),
            max_utilization_pct=_env_int(
                "ANNOTATION_GPU_MAX_UTILIZATION", 80, 0, 100
            ),
            poll_interval_sec=_env_float(
                "ANNOTATION_GPU_POLL_INTERVAL_SEC", 5.0, 0.5, 300.0
            ),
            status_log_interval_sec=_env_float(
                "ANNOTATION_GPU_STATUS_LOG_INTERVAL_SEC", 60.0, 1.0, 3600.0
            ),
            slot_memory_fraction=_env_float(
                "ANNOTATION_GPU_SLOT_MEMORY_FRACTION", 0.5, 0.05, 1.0
            ),
            max_slots_per_gpu=_env_int(
                "ANNOTATION_GPU_MAX_SLOTS_PER_DEVICE", 2, 1, 16
            ),
            logger=logger,
        )

    def _log(self, message: str, level: str = "INFO") -> None:
        if self._logger is not None:
            self._logger(message, phase="annotation", level=level)

    def _read_all(self) -> Tuple[List[GpuSnapshot], List[str]]:
        snapshots = []
        errors = []
        for gpu_id in self.gpu_ids:
            try:
                snapshots.append(self._snapshot_reader(gpu_id))
            except Exception as exc:
                errors.append(
                    f"GPU{gpu_id}(NVML error={type(exc).__name__}: {exc})"
                )
        return snapshots, errors

    def _summary(
        self,
        snapshots: Iterable[GpuSnapshot],
        errors: Iterable[str],
    ) -> str:
        states = []
        for snapshot in snapshots:
            gpu_id = snapshot["gpu_id"]
            reserved = self._reserved.get(gpu_id, 0)
            required = self._required_free_mib(snapshot)
            states.append(
                f"GPU{gpu_id}(free={snapshot['free_mib']}MiB,"
                f"required={required}MiB,"
                f"util={snapshot['utilization_pct']}%,"
                f"proc={snapshot['process_count']},"
                f"slots={reserved}/{self.max_slots_per_gpu})"
            )
        states.extend(errors)
        return "; ".join(states) or "no readable GPU"

    def acquire(self, label: str = "") -> int:
        deadline = time.monotonic() + self.wait_timeout_sec
        next_status_log = 0.0

        with self._condition:
            while True:
                snapshots, errors = self._read_all()
                candidates = []
                for snapshot in snapshots:
                    gpu_id = int(snapshot["gpu_id"])
                    if self._reserved.get(gpu_id, 0) >= self.max_slots_per_gpu:
                        continue
                    if snapshot["free_mib"] < self._required_free_mib(snapshot):
                        continue
                    if snapshot["utilization_pct"] > self.max_utilization_pct:
                        continue
                    candidates.append(snapshot)

                if candidates:
                    selected = max(
                        candidates,
                        key=lambda item: (
                            -self._reserved.get(int(item["gpu_id"]), 0),
                            item["free_mib"],
                            -item["utilization_pct"],
                            -item["process_count"],
                            -item["gpu_id"],
                        ),
                    )
                    gpu_id = int(selected["gpu_id"])
                    self._reserved[gpu_id] = self._reserved.get(gpu_id, 0) + 1
                    device_reserved = self._reserved[gpu_id]
                    total_reserved = sum(self._reserved.values())
                    self._log(
                        f"GPU {gpu_id} selected by NVML: {label} "
                        f"(free={selected['free_mib']}MiB, "
                        f"required={self._required_free_mib(selected)}MiB, "
                        f"used={selected['used_mib']}MiB, "
                        f"util={selected['utilization_pct']}%, "
                        f"proc={selected['process_count']}, "
                        f"device_slots={device_reserved}/{self.max_slots_per_gpu}, "
                        f"total_reserved={total_reserved}/{self.total_slot_capacity})"
                    )
                    return gpu_id

                now = time.monotonic()
                remaining = deadline - now
                summary = self._summary(snapshots, errors)
                if remaining <= 0:
                    message = (
                        f"GPU slot wait timed out ({self.wait_timeout_sec}s): {label}; "
                        f"required=free>=max({self.min_free_memory_mib}MiB,"
                        f"{self.slot_memory_fraction:.0%} total),"
                        f"util<={self.max_utilization_pct}%,"
                        f"slots<{self.max_slots_per_gpu}; {summary}"
                    )
                    self._log(message, level="ERROR")
                    raise GpuSlotTimeout(message)

                if now >= next_status_log:
                    self._log(
                        f"waiting for NVML GPU slot: {label} "
                        f"(required=free>=max({self.min_free_memory_mib}MiB,"
                        f"{self.slot_memory_fraction:.0%} total),"
                        f"util<={self.max_utilization_pct}%, "
                        f"slots<{self.max_slots_per_gpu}, "
                        f"timeout_remaining={int(remaining)}s); {summary}"
                    )
                    next_status_log = now + self.status_log_interval_sec

                self._condition.wait(
                    timeout=min(self.poll_interval_sec, remaining)
                )

    def release(self, gpu_id: int, label: str = "") -> None:
        with self._condition:
            device_reserved = self._reserved.get(gpu_id, 0)
            if device_reserved <= 0:
                self._log(
                    f"duplicate GPU {gpu_id} release ignored: {label}",
                    level="ERROR",
                )
                return
            self._reserved[gpu_id] = device_reserved - 1
            device_reserved = self._reserved[gpu_id]
            total_reserved = sum(self._reserved.values())
            self._condition.notify_all()

        self._log(
            f"GPU {gpu_id} released: {label} "
            f"(device_slots={device_reserved}/{self.max_slots_per_gpu}, "
            f"total_reserved={total_reserved}/{self.total_slot_capacity})"
        )


_pool_lock = threading.Lock()
_pool: Optional[NvmlGpuSlotPool] = None
_pool_signature: Optional[Tuple[Tuple[int, ...], int]] = None


def get_gpu_slot_pool(
    gpu_ids: Sequence[int],
    wait_timeout_sec: int,
    logger: Optional[LogWriter] = None,
) -> NvmlGpuSlotPool:
    """Return one scheduler shared by every annotation request in this process."""
    global _pool, _pool_signature
    signature = (tuple(gpu_ids), wait_timeout_sec)

    if _pool is not None:
        if _pool_signature != signature:
            raise RuntimeError(
                "GPU scheduler configuration changed after initialization; "
                "restart the API container"
            )
        return _pool

    with _pool_lock:
        if _pool is None:
            _pool = NvmlGpuSlotPool.from_env(
                gpu_ids=gpu_ids,
                wait_timeout_sec=wait_timeout_sec,
                logger=logger,
            )
            _pool_signature = signature
    return _pool

