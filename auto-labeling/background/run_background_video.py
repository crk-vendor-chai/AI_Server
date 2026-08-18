#!/usr/bin/env python3
"""Annotate a MinIO background ZIP with hand-only DINO + SAM3 video tracking."""

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


def prepare_input(
    records: list[dict],
    final_root: Path,
    input_dir: Path,
) -> list[tuple[str, str]]:
    """Create an ordered hard-link sequence and return temporary-to-final names."""
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    mapping: list[tuple[str, str]] = []
    for index, record in enumerate(records, start=1):
        output_name = record["output_name"]
        source = final_root / record["output_camera"] / output_name
        temporary_name = f"{index:08d}{source.suffix.lower()}"
        destination = input_dir / temporary_name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        mapping.append((Path(temporary_name).stem, output_name))
    return mapping


def merge_outputs(
    mapping: list[tuple[str, str]],
    clip_work: Path,
    camera_work: Path,
) -> None:
    for directory in ("labels", "labels_seg", "visualized"):
        (camera_work / directory).mkdir(parents=True, exist_ok=True)

    debug_by_name: dict[str, dict] = {}
    debug_path = clip_work / "detections.jsonl"
    if debug_path.is_file():
        for line in debug_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                row = json.loads(line)
                debug_by_name[Path(row["image"]).stem] = row

    merged_debug = camera_work / "detections.jsonl"
    with merged_debug.open("a", encoding="utf-8") as debug_out:
        for temporary_stem, output_name in mapping:
            output_stem = Path(output_name).stem
            for directory in ("labels", "labels_seg"):
                source = clip_work / directory / f"{temporary_stem}.txt"
                destination = camera_work / directory / f"{output_stem}.txt"
                if not source.is_file():
                    raise RuntimeError(f"missing output: {source}")
                shutil.copy2(source, destination)
            visualized = next(
                iter(sorted((clip_work / "visualized").glob(f"{temporary_stem}.*"))),
                None,
            )
            if visualized is None:
                raise RuntimeError(f"missing visualization: {temporary_stem}")
            shutil.copy2(visualized, camera_work / "visualized" / output_name)
            row = debug_by_name.get(temporary_stem)
            if row is not None:
                row["image"] = output_name
                debug_out.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minio-object",
        default="chaiimage/etc/background/Freeze_background.zip",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sam3_with_dino" / "background",
    )
    parser.add_argument("--dataset-name", default="Freeze_HAND_BACKGROUND_VIDEO")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--dino-batch", type=int, default=4)
    parser.add_argument("--hand-box-threshold", type=float, default=0.25)
    parser.add_argument("--video-hand-seed-threshold", type=float, default=0.30)
    parser.add_argument("--video-min-output-score", type=float, default=0.40)
    args = parser.parse_args()

    if str(args.physical_gpu) != "1":
        raise ValueError("this background run is intentionally restricted to physical GPU 1")

    final_root = args.runs_root / args.dataset_name
    work_root = args.runs_root / "_work" / args.dataset_name
    manifest_path = work_root / "source_manifest.json"
    if args.overwrite:
        shutil.rmtree(final_root, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)
    final_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # The Docker command exposes only physical GPU 1. CUDA renumbers that
    # single visible device to ordinal 0 inside the container.
    if env.get("NVIDIA_VISIBLE_DEVICES") not in {"1", "GPU-1"}:
        raise RuntimeError("run this script in a container exposing only physical GPU 1")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT / "auto-labeling" / "sam3_with_dino")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    if not args.skip_prepare:
        run_logged(
            [
                args.python_bin,
                str(SCRIPT_DIR / "prepare_background.py"),
                "--minio-object",
                args.minio_object,
                "--output-root",
                str(final_root),
                "--manifest",
                str(manifest_path),
                "--camera-map",
                "cam_0=cam0",
                "cam_2=cam2",
                "--mapped-only",
            ],
            work_root / "prepare.log",
            env,
        )
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in manifest["images"]:
        grouped[(record["clip"], record["output_camera"])].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["source_path"])

    camera_work_roots = {
        camera: work_root / camera for camera in sorted(manifest["cameras"])
    }
    for camera_work in camera_work_roots.values():
        shutil.rmtree(camera_work, ignore_errors=True)
        camera_work.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    ordered_groups = sorted(grouped.items(), key=lambda item: item[0])
    for number, ((clip, camera), records) in enumerate(ordered_groups, start=1):
        clip_work = work_root / "clips" / clip / camera
        shutil.rmtree(clip_work, ignore_errors=True)
        clip_work.mkdir(parents=True, exist_ok=True)
        mapping = prepare_input(records, final_root, clip_work / "input")
        print(
            f"[start {number}/{len(ordered_groups)}] clip={clip} camera={camera} "
            f"images={len(records)} physical_gpu={args.physical_gpu}",
            flush=True,
        )
        command = [
            args.python_bin,
            str(VIDEO_ANNOTATOR),
            "--hand_only",
            "--img_dir",
            str(clip_work / "input"),
            "--out_label_dir",
            str(clip_work / "labels"),
            "--out_seg_label_dir",
            str(clip_work / "labels_seg"),
            "--out_visualized_dir",
            str(clip_work / "visualized"),
            "--out_debug_jsonl",
            str(clip_work / "detections.jsonl"),
            "--summary_path",
            str(clip_work / "summary.txt"),
            "--hand_prompts",
            "human hand|left hand|right hand|palm|fingers",
            "--sam3_hand_prompt",
            "human hand and fingers",
            "--hand_box_threshold",
            str(args.hand_box_threshold),
            "--video_hand_seed_threshold",
            str(args.video_hand_seed_threshold),
            "--video_min_output_score",
            str(args.video_min_output_score),
            "--hand_recovery_max_anchors",
            "0",
            "--dino_batch",
            str(args.dino_batch),
            "--dino_autocast_dtype",
            "bfloat16",
            "--segmentation_fallback",
            "box",
            "--retain_all_frames",
        ]
        run_logged(command, clip_work / "run.log", env)
        merge_outputs(mapping, clip_work, camera_work_roots[camera])
        summary = {
            "clip": clip,
            "camera": camera,
            "images": len(records),
            "summary": str(clip_work / "summary.txt"),
        }
        summaries.append(summary)
        print(
            f"[done {number}/{len(ordered_groups)}] clip={clip} camera={camera}",
            flush=True,
        )

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
        work_root / "export.log",
        env,
    )
    payload = {
        "source": args.minio_object,
        "method": "grounding_dino_seeded_sam3_video_hand_only",
        "physical_gpu": args.physical_gpu,
        "clips": len({clip for clip, _ in grouped}),
        "groups": summaries,
        "images": sum(len(records) for records in grouped.values()),
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
