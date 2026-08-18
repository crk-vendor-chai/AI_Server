"""product/created 등 요청(IF_SYSID) 단위 pipeline 로그."""
from __future__ import annotations

import contextvars
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.join(_APP_ROOT, "logs", "requests")

_request_ctx: contextvars.ContextVar[Optional["RequestLogContext"]] = contextvars.ContextVar(
    "request_log_ctx", default=None
)
_print_lock = threading.Lock()


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "unknown").strip()) or "unknown"


@dataclass
class RequestLogContext:
    """한 번의 API 요청(created / annotation/completed 등) 추적용."""

    request_id: str
    event: str
    division_idx: str = ""
    is_cold: Optional[bool] = None
    if_sysid: str = ""
    if_id: str = ""
    log_dir: str = ""
    pipeline_log: str = field(init=False, default="")
    jobs_dir: str = field(init=False, default="")

    def __post_init__(self) -> None:
        root = os.getenv("PNT_REQUEST_LOG_DIR", _DEFAULT_ROOT).strip() or _DEFAULT_ROOT
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = (
            f"{_safe_name(self.division_idx or 'unknown')}_"
            f"{_safe_name(self.if_sysid or self.request_id)}_{ts}"
        )
        self.log_dir = os.path.join(root, folder)
        self.pipeline_log = os.path.join(self.log_dir, "pipeline.log")
        self.jobs_dir = os.path.join(self.log_dir, "jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)

    def bind(self) -> "RequestLogContext":
        _request_ctx.set(self)
        return self

    def job_log_dir(self, job_type: str) -> str:
        path = os.path.join(self.jobs_dir, job_type)
        os.makedirs(path, exist_ok=True)
        return path


def get_request_context() -> Optional[RequestLogContext]:
    return _request_ctx.get()


def bind_request_context(ctx: Optional[RequestLogContext]) -> None:
    _request_ctx.set(ctx)


def init_request_log(
    *,
    event: str,
    if_sysid: str = "",
    if_id: str = "",
    division_idx: str = "",
    is_cold: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
    log_dir: str = "",
) -> RequestLogContext:
    request_id = (if_sysid or if_id or division_idx or "unknown").strip() or "unknown"
    ctx = RequestLogContext(
        request_id=request_id,
        event=event,
        division_idx=division_idx,
        is_cold=is_cold,
        if_sysid=if_sysid,
        if_id=if_id,
    )
    if log_dir:
        ctx.log_dir = log_dir
        ctx.pipeline_log = os.path.join(log_dir, "pipeline.log")
        ctx.jobs_dir = os.path.join(log_dir, "jobs")
        os.makedirs(ctx.jobs_dir, exist_ok=True)
    ctx.bind()
    header = (
        f"===== REQUEST {event.upper()} {datetime.now().isoformat(timespec='seconds')} =====\n"
        f"request_id={request_id}\n"
        f"IF_ID={if_id}\n"
        f"IF_SYSID={if_sysid}\n"
        f"division_idx={division_idx}\n"
        f"is_cold={is_cold}\n"
        f"log_dir={ctx.log_dir}\n"
    )
    if extra:
        for key, value in extra.items():
            header += f"{key}={value}\n"
    header += "================================================================\n"
    with open(ctx.pipeline_log, "a", encoding="utf-8", buffering=1) as fp:
        fp.write(header)
    pipeline_log(f"request log initialized: {ctx.log_dir}", phase=event)
    return ctx


def resume_request_log(
    run: dict,
    *,
    event: str,
    if_sysid: str = "",
    if_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> RequestLogContext:
    """PipelineRun에 저장된 log_dir로 같은 created run 로그에 이어 씀."""
    return init_request_log(
        event=event,
        if_sysid=if_sysid or str(run.get("run_id") or ""),
        if_id=if_id or str(run.get("if_id") or ""),
        division_idx=str(run.get("division_idx") or ""),
        is_cold=run.get("is_cold"),
        log_dir=str(run.get("log_dir") or ""),
        extra=extra,
    )


def pipeline_log(message: str, *, phase: str = "pipeline", level: str = "INFO") -> None:
    """요청 pipeline.log + stdout(api.log) 동시 기록."""
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] [{level}] [{phase}] {message}\n"
    console = f"[{phase}] {message}"
    ctx = _request_ctx.get()
    with _print_lock:
        print(console, flush=True)
        if ctx and ctx.pipeline_log:
            with open(ctx.pipeline_log, "a", encoding="utf-8", buffering=1) as fp:
                fp.write(line)
