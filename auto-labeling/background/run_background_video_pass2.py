#!/usr/bin/env python3
"""Run fill-ratio-gated pass 2 for a completed background video pass 1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from asis_pipeline import _write_fill_file
from run_background_video import merge_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
VIDEO_ANNOTATOR = PROJECT_ROOT / "auto-labeling" / "sam3_with_dino" / "video_annotate.py"


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {log_path}")


def mapping_for(records: list[dict]) -> list[tuple[str, str]]:
    return [
        (f"{index:08d}", record["output_name"])
        for index, record in enumerate(records, start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sam3_with_dino" / "background",
    )
    parser.add_argument("--dataset-name", default="Freeze_HAND_BACKGROUND_VIDEO")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--fill-threshold", type=float, default=0.45)
    args = parser.parse_args()

    if str(args.physical_gpu) != "1":
        raise ValueError("this background run is intentionally restricted to physical GPU 1")
    if not 0.0 <= args.fill_threshold <= 1.0:
        raise ValueError("--fill-threshold must be between 0 and 1")

    final_root = args.runs_root / args.dataset_name
    work_root = args.runs_root / "_work" / args.dataset_name
    manifest_path = work_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    env = os.environ.copy()
    if env.get("NVIDIA_VISIBLE_DEVICES") not in {"1", "GPU-1"}:
        raise RuntimeError("run this script in a container exposing only physical GPU 1")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT / "auto-labeling" / "sam3_with_dino"),
            str(SCRIPT_DIR),
            env.get("PYTHONPATH", ""),
        ]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in manifest["images"]:
        grouped[(record["clip"], record["output_camera"])].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["source_path"])

    expected = len(grouped)
    pass1_complete = sum(
        (work_root / "clips" / clip / camera / "summary.txt").is_file()
        for clip, camera in grouped
    )
    if pass1_complete != expected:
        raise RuntimeError(f"pass 1 incomplete: {pass1_complete}/{expected}")

    final_camera_work = {
        camera: work_root / camera for camera in sorted(manifest["cameras"])
    }
    for camera_work in final_camera_work.values():
        shutil.rmtree(camera_work, ignore_errors=True)
        camera_work.mkdir(parents=True, exist_ok=True)

    fill_counts: dict[str, int] = {}
    summaries: list[dict] = []
    ordered_groups = sorted(grouped.items(), key=lambda item: item[0])
    for number, ((clip, camera), records) in enumerate(ordered_groups, start=1):
        pass1 = work_root / "clips" / clip / camera
        pass2 = work_root / "clips_pass2" / clip / camera
        shutil.rmtree(pass2, ignore_errors=True)
        pass2.mkdir(parents=True, exist_ok=True)
        fill_path = work_root / "fill" / clip / f"{camera}.json"
        fill_count = _write_fill_file(
            pass1 / "detections.jsonl",
            pass1 / "labels_seg",
            pass1 / "input",
            fill_path,
        )
        fill_counts[f"{clip}/{camera}"] = fill_count
        print(
            f"[pass2 {number}/{expected}] clip={clip} camera={camera} "
            f"images={len(records)} fill_frames={fill_count} threshold={args.fill_threshold}",
            flush=True,
        )
        command = [
            args.python_bin,
            str(VIDEO_ANNOTATOR),
            "--hand_only",
            "--img_dir",
            str(pass1 / "input"),
            "--out_label_dir",
            str(pass2 / "labels"),
            "--out_seg_label_dir",
            str(pass2 / "labels_seg"),
            "--out_visualized_dir",
            str(pass2 / "visualized"),
            "--out_debug_jsonl",
            str(pass2 / "detections.jsonl"),
            "--summary_path",
            str(pass2 / "summary.txt"),
            "--hand_prompts",
            "human hand|left hand|right hand|palm|fingers",
            "--sam3_hand_prompt",
            "human hand and fingers",
            "--hand_box_threshold",
            "0.25",
            "--video_hand_seed_threshold",
            "0.30",
            "--video_min_output_score",
            "0.40",
            "--hand_recovery_max_anchors",
            "0",
            "--hand_seed_dual",
            "--hand_seed_halves",
            "2",
            "--hand_seed_fill_file",
            str(fill_path),
            "--hand_seed_fill_min",
            str(args.fill_threshold),
            "--dino_cache",
            str(pass1 / "detections.jsonl"),
            "--dino_batch",
            "4",
            "--dino_autocast_dtype",
            "bfloat16",
            "--segmentation_fallback",
            "box",
            "--retain_all_frames",
        ]
        run_logged(command, pass2 / "run.log", env)
        merge_outputs(mapping_for(records), pass2, final_camera_work[camera])
        summaries.append(
            {"clip": clip, "camera": camera, "images": len(records), "fill_frames": fill_count}
        )
        print(f"[pass2 done {number}/{expected}] clip={clip} camera={camera}", flush=True)

    cameras = sorted(manifest["cameras"], key=lambda name: int(name[3:]))
    run_logged(
        [
            args.python_bin,
            str(SCRIPT_DIR / "export_coco.py"),
            "--final-root",
            str(final_root),
            "--work-root",
            str(work_root),
            "--cameras",
            *cameras,
            "--output",
            str(final_root / "annotations_hand_background.json"),
        ],
        work_root / "export_pass2.log",
        env,
    )
    payload = {
        "source": manifest["source"],
        "method": "grounding_dino_seeded_sam3_video_hand_only_fill_two_pass",
        "physical_gpu": args.physical_gpu,
        "fill_threshold": args.fill_threshold,
        "pass1_sequences": expected,
        "pass2_sequences": len(summaries),
        "images": sum(item["images"] for item in summaries),
        "fill_frames": fill_counts,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (work_root / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
