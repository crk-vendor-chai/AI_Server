#!/usr/bin/env python3
"""Annotate zero, one, or two hands with Grounding DINO and SAM3."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

SAM3_DINO_DIR = Path(__file__).resolve().parents[1] / "sam3_with_dino"
sys.path.insert(0, str(SAM3_DINO_DIR))

from auto_annotate import (  # noqa: E402
    HAND_CLASS_ID,
    Candidate,
    FinalObject,
    box_area_similarity,
    box_center_distance_norm,
    box_iou,
    build_sam3_processor,
    detect_with_dino,
    draw_visualized,
    image_files,
    load_grounding_dino,
    nms,
    read_hf_tokens_from_env_file,
    refine_with_sam3,
    split_prompts,
    write_detect_label,
    write_jsonl,
    write_segment_label,
)


def clip_name(image_path: Path) -> str:
    stem = image_path.stem
    if "_frame_" in stem:
        return stem.split("_frame_", 1)[0]
    return stem.split("_", 1)[0]


def merged_image_sort_key(image_path: Path) -> tuple[int, int, str]:
    parts = image_path.stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1]), image_path.name
    return sys.maxsize, sys.maxsize, image_path.name


def temporal_compatibility(
    box: tuple[float, float, float, float],
    previous_boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> float:
    best = 0.0
    for previous in previous_boxes:
        iou = box_iou(box, previous)
        center = math.exp(-6.0 * box_center_distance_norm(box, previous, width, height))
        area = box_area_similarity(box, previous)
        best = max(best, 0.65 * iou + 0.25 * center + 0.10 * area)
    return best


def box_overlap_over_smaller(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, min(area_a, area_b))


def is_low_confidence_corner_candidate(
    candidate: Candidate,
    width: int,
    height: int,
    margin: float,
    min_score: float,
) -> bool:
    x1, y1, x2, y2 = candidate.box_xyxy
    touches_left = x1 <= width * margin
    touches_right = x2 >= width * (1.0 - margin)
    touches_top = y1 <= height * margin
    touches_top_corner = (touches_left or touches_right) and touches_top
    return touches_top_corner and candidate.score < min_score


def hand_shape_ok(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    max_width_ratio: float = 1.0,
    max_height_ratio: float = 1.0,
) -> bool:
    if width <= 0 or height <= 0 or not all(math.isfinite(value) for value in box):
        return False
    x1, y1, x2, y2 = box
    clipped_width = max(0.0, min(float(width), x2) - max(0.0, x1))
    clipped_height = max(0.0, min(float(height), y2) - max(0.0, y1))
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return False
    area_ratio = clipped_width * clipped_height / float(width * height)
    aspect = clipped_width / clipped_height
    return (
        min_area <= area_ratio <= max_area
        and min_aspect <= aspect <= max_aspect
        and clipped_width / width <= max_width_ratio
        and clipped_height / height <= max_height_ratio
    )


def select_hands(
    candidates: list[Candidate],
    previous_boxes: list[tuple[float, float, float, float]],
    previous_age: int,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[list[Candidate], list[dict]]:
    filtered = [
        candidate
        for candidate in candidates
        if hand_shape_ok(
            candidate.box_xyxy,
            width,
            height,
            args.min_hand_area,
            args.max_hand_area,
            args.min_hand_aspect,
            args.max_hand_aspect,
            args.max_hand_width,
            args.max_hand_height,
        )
    ]
    filtered = nms(filtered, args.nms_iou)

    ranked: list[tuple[float, Candidate, float]] = []
    decay = args.temporal_decay ** max(0, previous_age)
    for candidate in filtered:
        compatibility = temporal_compatibility(candidate.box_xyxy, previous_boxes, width, height)
        rank = candidate.score + args.temporal_weight * compatibility * decay
        ranked.append((rank, candidate, compatibility))
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected: list[Candidate] = []
    debug: list[dict] = []
    for rank, candidate, compatibility in ranked:
        corner_rejected = (
            is_low_confidence_corner_candidate(
                candidate,
                width,
                height,
                args.corner_margin,
                args.corner_min_score,
            )
            and compatibility < args.corner_min_temporal_compatibility
        )
        accepted = not corner_rejected and (candidate.score >= args.min_output_score or (
            previous_boxes
            and candidate.score >= args.hand_box_threshold
            and compatibility >= args.min_temporal_compatibility
        ))
        duplicate = any(
            box_iou(candidate.box_xyxy, item.box_xyxy) >= args.final_nms_iou
            or box_overlap_over_smaller(candidate.box_xyxy, item.box_xyxy) >= args.containment_overlap
            for item in selected
        )
        accepted = accepted and not duplicate
        debug.append({
            "box_xyxy": list(candidate.box_xyxy),
            "score": candidate.score,
            "temporal_compatibility": compatibility,
            "rank": rank,
            "accepted": bool(accepted),
            "duplicate": bool(duplicate),
            "corner_rejected": bool(corner_rejected),
        })
        if not accepted:
            continue
        selected.append(candidate)
        if len(selected) >= args.max_hands:
            break
    return selected, debug


def deduplicate_final(
    objects: list[FinalObject],
    iou_threshold: float,
    containment_overlap: float,
) -> list[FinalObject]:
    kept: list[FinalObject] = []
    for obj in sorted(objects, key=lambda item: item.score, reverse=True):
        if any(
            box_iou(obj.box_xyxy, other.box_xyxy) >= iou_threshold
            or box_overlap_over_smaller(obj.box_xyxy, other.box_xyxy) >= containment_overlap
            for other in kept
        ):
            continue
        kept.append(obj)
    return sorted(kept, key=lambda item: item.box_xyxy[0])


@torch.inference_mode()
def annotate(args: argparse.Namespace) -> dict:
    files = sorted(image_files(args.img_dir), key=merged_image_sort_key)
    if args.start_index:
        files = files[args.start_index:]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"no images found: {args.img_dir}")

    for directory in (
        args.output_root / "labels",
        args.output_root / "labels_seg",
        args.output_root / "visualized",
        args.output_root / "debug",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    debug_path = args.output_root / "debug" / "detections.jsonl"
    if debug_path.exists():
        debug_path.unlink()

    print(f"[info] loading Grounding DINO: {args.dino_model}", flush=True)
    dino_processor, dino_model = load_grounding_dino(args.dino_model, args.dino_device)
    print("[info] loading SAM3 image model", flush=True)
    sam3_processor = build_sam3_processor(args.sam3_device, args.sam3_confidence)

    previous_boxes: list[tuple[float, float, float, float]] = []
    previous_age = 0
    previous_clip = ""
    empty_frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    total_boxes = 0
    mask_boxes = 0
    started = time.time()

    for index, image_path in enumerate(files):
        current_clip = clip_name(image_path)
        if current_clip != previous_clip:
            previous_boxes = []
            previous_age = 0
            previous_clip = current_clip

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        candidates = detect_with_dino(
            image,
            dino_processor,
            dino_model,
            args.dino_device,
            args.hand_prompts,
            HAND_CLASS_ID,
            "hand",
            args.hand_box_threshold,
            args.text_threshold,
            args.max_hand_candidates,
        )
        selected, selection_debug = select_hands(
            candidates,
            previous_boxes,
            previous_age,
            width,
            height,
            args,
        )

        final_objects: list[FinalObject] = []
        if selected:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(args.sam3_device).startswith("cuda"),
            ):
                state = sam3_processor.set_image(image)
            for candidate in selected:
                refined = refine_with_sam3(image, state, sam3_processor, candidate, args)
                if hand_shape_ok(
                    refined.box_xyxy,
                    width,
                    height,
                    args.min_hand_area,
                    args.max_hand_area,
                    args.min_hand_aspect,
                    args.max_hand_aspect,
                    args.max_hand_width,
                    args.max_hand_height,
                ):
                    final_objects.append(refined)
                else:
                    final_objects.append(FinalObject(
                        candidate.class_id,
                        candidate.class_name,
                        candidate.box_xyxy,
                        candidate.score,
                        candidate.prompt,
                    ))
            final_objects = deduplicate_final(
                final_objects,
                args.final_nms_iou,
                args.containment_overlap,
            )[:args.max_hands]

        detect_path = args.output_root / "labels" / f"{image_path.stem}.txt"
        segment_path = args.output_root / "labels_seg" / f"{image_path.stem}.txt"
        visualized_path = args.output_root / "visualized" / image_path.name
        write_detect_label(detect_path, final_objects, width, height)
        write_segment_label(
            segment_path,
            final_objects,
            width,
            height,
            args.segmentation_fallback,
        )
        draw_visualized(image, final_objects, visualized_path)

        count = len(final_objects)
        empty_frames += count == 0
        one_hand_frames += count == 1
        two_hand_frames += count == 2
        total_boxes += count
        mask_boxes += sum(bool(obj.mask_valid) for obj in final_objects)

        if final_objects:
            previous_boxes = [obj.box_xyxy for obj in final_objects]
            previous_age = 0
        elif previous_boxes:
            previous_age += 1
            if previous_age > args.temporal_max_gap:
                previous_boxes = []
                previous_age = 0

        write_jsonl(debug_path, {
            "image": image_path.name,
            "clip": current_clip,
            "candidates": [asdict(candidate) for candidate in candidates],
            "selection": selection_debug,
            "final": [asdict(obj) for obj in final_objects],
        })
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(files):
            print(
                f"[progress {index + 1}/{len(files)}] {image_path.name} "
                f"hands={count} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    summary = {
        "backend": "grounding_dino_plus_sam3_image",
        "class": "hand",
        "max_hands": args.max_hands,
        "images": len(files),
        "empty_frames": empty_frames,
        "one_hand_frames": one_hand_frames,
        "two_hand_frames": two_hand_frames,
        "total_boxes": total_boxes,
        "sam3_mask_boxes": mask_boxes,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (args.output_root / "summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    del dino_model, dino_processor, sam3_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(summary), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--hand-prompts", default="complete human hand|hand from wrist to fingertips|left human hand|right human hand|grasping human hand")
    parser.add_argument("--hand-box-threshold", type=float, default=0.24)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--min-output-score", type=float, default=0.32)
    parser.add_argument("--max-hand-candidates", type=int, default=8)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--final-nms-iou", type=float, default=0.70)
    parser.add_argument("--containment-overlap", type=float, default=0.80)
    parser.add_argument("--corner-margin", type=float, default=0.01)
    parser.add_argument("--corner-min-score", type=float, default=0.55)
    parser.add_argument("--corner-min-temporal-compatibility", type=float, default=0.20)
    parser.add_argument("--min-hand-area", type=float, default=0.0015)
    parser.add_argument("--max-hand-area", type=float, default=0.20)
    parser.add_argument("--min-hand-aspect", type=float, default=0.18)
    parser.add_argument("--max-hand-aspect", type=float, default=4.5)
    parser.add_argument("--max-hand-width", type=float, default=0.70)
    parser.add_argument("--max-hand-height", type=float, default=0.80)
    parser.add_argument("--temporal-weight", type=float, default=0.20)
    parser.add_argument("--temporal-decay", type=float, default=0.80)
    parser.add_argument("--temporal-max-gap", type=int, default=4)
    parser.add_argument("--min-temporal-compatibility", type=float, default=0.20)
    parser.add_argument("--sam3-confidence", type=float, default=0.35)
    parser.add_argument("--sam3-autocast-dtype", choices=["bfloat16", "float16", "none"], default="bfloat16")
    parser.add_argument("--sam3-iou-weight", type=float, default=0.30)
    parser.add_argument("--sam3-area-weight", type=float, default=0.15)
    parser.add_argument("--sam3-min-prompt-iou", type=float, default=0.15)
    parser.add_argument("--sam3-min-area-ratio", type=float, default=0.15)
    parser.add_argument("--sam3-max-area-ratio", type=float, default=3.0)
    parser.add_argument("--polygon-epsilon", type=float, default=0.003)
    parser.add_argument("--min-polygon-points", type=int, default=3)
    parser.add_argument("--segmentation-fallback", choices=["skip", "box"], default="box")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    read_hf_tokens_from_env_file(Path(__file__).resolve().parents[2] / ".env")
    args.hand_prompts = split_prompts(args.hand_prompts)
    annotate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
