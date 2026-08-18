"""annotation / segment 등 subprocess job 로그를 0_pnt/logs/ 아래에 저장."""
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JOB_LOG_ROOT = os.path.join(_APP_ROOT, "logs")


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "unknown").strip()) or "unknown"


def make_job_log_path(
    job_type: str,
    name: str,
    *,
    log_dir: Optional[str] = None,
) -> str:
    """
    job_type: annotation | segment | training | deploy 등
    name: 보통 foldername 또는 division_idx
    log_dir: 지정 시 해당 디렉터리에 저장 (요청별 logs/requests/.../jobs/annotation 등)
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{_safe_name(name)}_{ts}.log"
        return os.path.join(log_dir, filename)

    root = os.getenv("PNT_JOB_LOG_DIR", _DEFAULT_JOB_LOG_ROOT).strip() or _DEFAULT_JOB_LOG_ROOT
    default_dir = os.path.join(root, job_type)
    os.makedirs(default_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_name(name)}_{ts}.log"
    return os.path.join(default_dir, filename)


def run_shell_job(
    cmd: List[str],
    *,
    job_type: str,
    log_name: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
    log_prefix: str = "",
    job_log_dir: Optional[str] = None,
    tee_stdout: Optional[bool] = None,
) -> subprocess.CompletedProcess:
    """
    bash/python job 실행. stdout·stderr는 job 전용 log 파일에 저장.
    tee_stdout=True(기본)면 api.log에도 출력. annotation 등은 False 권장.
    """
    if tee_stdout is None:
        tee_stdout = os.getenv("PNT_JOB_LOG_TEE_STDOUT", "1").strip().lower() in (
            "1", "true", "yes",
        )

    log_path = make_job_log_path(job_type, log_name, log_dir=job_log_dir)
    if env is not None:
        env = dict(env)
        env["PNT_JOB_LOG_PATH"] = log_path
        env["PNT_JOB_LOG_DIR"] = os.path.dirname(log_path)
        env["PNT_JOB_LOG_BASENAME"] = os.path.splitext(os.path.basename(log_path))[0]
    prefix = log_prefix or f"[{job_type}_job]"
    print(f"{prefix} log file: {log_path}" + ("" if tee_stdout else " (api.log tee off)"))

    header = (
        f"===== {job_type.upper()} JOB START {datetime.now().isoformat(timespec='seconds')} =====\n"
        f"cmd: {' '.join(cmd)}\n"
        f"cwd: {cwd or os.getcwd()}\n"
        "================================================================\n"
    )

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_fp:
        log_fp.write(header)
        log_fp.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if tee_stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                log_fp.write(line)
                log_fp.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            msg = f"{prefix} 타임아웃 (log: {log_path})\n"
            print(msg, file=sys.stderr)
            log_fp.write(msg)
            raise
        finally:
            footer = (
                f"\n===== {job_type.upper()} JOB END {datetime.now().isoformat(timespec='seconds')} "
                f"exit={proc.returncode} =====\n"
            )
            log_fp.write(footer)
            log_fp.flush()

    return subprocess.CompletedProcess(cmd, proc.returncode or 0)
