#!/usr/bin/env python3
"""Prepare and annotate a MinIO background ZIP on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        return subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env or os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minio-object", default="chaiimage/etc/background/20260722_12.zip")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs" / "sam3_with_dino" / "background")
    parser.add_argument("--dataset-name", default="20260722_12_HAND_BACKGROUND")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--camera-map", nargs="+", default=[])
    parser.add_argument("--mapped-only", action="store_true")
    parser.add_argument("--hand-box-threshold", type=float, default=0.24)
    parser.add_argument("--min-output-score", type=float, default=0.32)
    parser.add_argument("--max-hands", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    final_root = args.runs_root / args.dataset_name
    work_root = args.runs_root / "_work" / args.dataset_name
    manifest_path = work_root / "source_manifest.json"
    if args.overwrite:
        shutil.rmtree(final_root, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)
    final_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_cmd = [
            args.python_bin,
            str(SCRIPT_DIR / "prepare_background.py"),
            "--minio-object", args.minio_object,
            "--output-root", str(final_root),
            "--manifest", str(manifest_path),
        ]
        if args.camera_map:
            prepare_cmd.extend(["--camera-map", *args.camera_map])
        if args.mapped_only:
            prepare_cmd.append("--mapped-only")
        if run_logged(prepare_cmd, work_root / "prepare.log") != 0:
            print(f"[error] preparation failed: {work_root / 'prepare.log'}", file=sys.stderr)
            return 1

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cameras = manifest.get("cameras") or list(manifest.get("counts", {}))
    else:
        cameras = [
            path.name for path in final_root.iterdir()
            if path.is_dir() and path.name.startswith("cam") and path.name[3:].isdigit()
        ]
    cameras = sorted(cameras, key=lambda name: int(name[3:]))
    if not cameras:
        raise RuntimeError(f"no camera folders found: {final_root}")
    processes: list[tuple[str, str, subprocess.Popen, object]] = []
    for index, camera in enumerate(cameras):
        gpu = args.gpus[index % len(args.gpus)]
        camera_work = work_root / camera
        shutil.rmtree(camera_work, ignore_errors=True)
        camera_work.mkdir(parents=True, exist_ok=True)
        log_file = (camera_work / "run.log").open("w", encoding="utf-8")
        cmd = [
            args.python_bin,
            str(SCRIPT_DIR / "annotate_hands.py"),
            "--img-dir", str(final_root / camera),
            "--output-root", str(camera_work),
            "--hand-box-threshold", str(args.hand_box_threshold),
            "--min-output-score", str(args.min_output_score),
            "--max-hands", str(args.max_hands),
            "--segmentation-fallback", "box",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(SCRIPT_DIR.parents[0] / "sam3_with_dino") + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        log_file.write(" ".join(cmd) + f"\nCUDA_VISIBLE_DEVICES={gpu}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((camera, gpu, process, log_file))
        print(f"[start] {camera} gpu={gpu} pid={process.pid}", flush=True)

    failed = 0
    for camera, gpu, process, log_file in processes:
        returncode = process.wait()
        log_file.close()
        print(f"[done] {camera} gpu={gpu} returncode={returncode}", flush=True)
        failed += returncode != 0
    if failed:
        return 1

    export_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "export_coco.py"),
        "--final-root", str(final_root),
        "--cameras", *cameras,
        "--work-root", str(work_root),
        "--output", str(final_root / "annotations_hand_background.json"),
    ]
    if run_logged(export_cmd, work_root / "export.log") != 0:
        print(f"[error] COCO export failed: {work_root / 'export.log'}", file=sys.stderr)
        return 1

    summary = {
        "source": args.minio_object,
        "final_root": str(final_root),
        "work_root": str(work_root),
        "cameras": cameras,
        "gpus": args.gpus,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (work_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
