#!/usr/bin/env python3
"""Grounding DINO seeded SAM3 video tracking for YOLO annotation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import os
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_model

from auto_annotate import (
    HAND_CLASS_ID,
    Candidate,
    FinalObject,
    box_area,
    box_area_similarity,
    box_center_distance_norm,
    box_gap_norm,
    box_iou,
    cross_class_duplicate,
    detect_with_dino,
    draw_visualized,
    load_grounding_dino,
    mask_to_box,
    mask_to_polygon,
    nms,
    object_selection_score,
    product_prompts_from_folder,
    read_hf_tokens_from_env_file,
    relation_bonus,
    candidates_from_results,
    dino_autocast,
    grounding_text,
    select_final_candidates,
    set_dino_autocast,
    shape_ok,
    split_prompts,
    strip_folder_timestamp,
    write_detect_label,
    write_jsonl,
    write_segment_label,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
WHOLE_HAND_PROMPT_TOKENS = (
    "complete human hand",
    "whole human hand",
    "hand from wrist",
    "grasping hand",
    "grasping human hand",
    "human hand",
    "left hand",
    "right hand",
)
EXPLICIT_WHOLE_HAND_PROMPT_TOKENS = (
    "complete human hand",
    "whole human hand",
    "hand from wrist",
    "grasping hand",
    "grasping human hand",
)
HAND_DETAIL_PROMPT_TOKENS = ("finger", "palm", "knuckle")
SHOE_COMPETITOR_CLASS_ID = -100
ARM_COMPETITOR_CLASS_ID = -101


@dataclass
class FrameDetections:
    hand_candidates: list[Candidate]
    object_candidates: list[Candidate]
    best_hand: Candidate | None
    best_object: Candidate | None
    fallback_hand: Candidate | None
    fallback_object: Candidate | None
    shoe_candidates: list[Candidate] = field(default_factory=list)
    arm_candidates: list[Candidate] = field(default_factory=list)
    best_shoe: Candidate | None = None
    best_arm: Candidate | None = None
    foreground_mask: np.ndarray | None = None
    background_reference: str | None = None


@dataclass
class TrackResult:
    obj: FinalObject
    mask: np.ndarray | None
    source: str
    output_score: float


def image_files(img_dir: Path) -> list[Path]:
    source = img_dir
    direct = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not direct and (source / "images").is_dir():
        source = source / "images"
        direct = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    try:
        return sorted(direct, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(direct)


def autocast_for(device: str, dtype_name: str):
    if not str(device).startswith("cuda") or dtype_name == "none":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def role_shape_ok(candidate: Candidate, width: int, height: int, args: argparse.Namespace) -> bool:
    is_hand = candidate.class_id == HAND_CLASS_ID
    return shape_ok(
        candidate.box_xyxy,
        width,
        height,
        args.min_hand_area if is_hand else args.min_object_area,
        args.max_hand_area if is_hand else args.max_object_area,
        args.min_hand_aspect if is_hand else args.min_object_aspect,
        args.max_hand_aspect if is_hand else args.max_object_aspect,
        args.max_hand_width if is_hand else args.max_object_width,
        args.max_hand_height if is_hand else args.max_object_height,
    )


def filtered_candidates(
    candidates: list[Candidate],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[Candidate]:
    return nms(
        [candidate for candidate in candidates if role_shape_ok(candidate, width, height, args)],
        args.nms_iou,
    )



def load_background_references(paths: list[str]) -> list[tuple[str, np.ndarray]]:
    references: list[tuple[str, np.ndarray]] = []
    for raw_path in paths:
        path = Path(raw_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] background reference unreadable: {path}", file=sys.stderr)
            continue
        references.append((path.name, image))
    return references


def foreground_mask_for_image(
    image: Image.Image,
    references: list[tuple[str, np.ndarray]],
    pixel_threshold: float,
) -> tuple[np.ndarray | None, str | None]:
    if not references:
        return None, None

    current = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    current_blur = cv2.GaussianBlur(current, (5, 5), 0)
    ranked: list[tuple[float, str, np.ndarray]] = []
    for name, reference in references:
        resized = (
            cv2.resize(reference, (current.shape[1], current.shape[0]))
            if reference.shape[:2] != current.shape[:2]
            else reference
        )
        reference_blur = cv2.GaussianBlur(resized, (5, 5), 0)
        difference = cv2.absdiff(current_blur, reference_blur).max(axis=2)
        match_score = float(np.clip(difference, 0, 50).mean())
        ranked.append((match_score, name, difference))

    _, name, difference = min(ranked, key=lambda item: item[0])
    foreground = (difference >= float(pixel_threshold)).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    foreground = cv2.dilate(
        foreground,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return foreground.astype(bool), name


def box_foreground_fraction(
    foreground_mask: np.ndarray | None,
    box: tuple[float, float, float, float],
) -> float:
    if foreground_mask is None:
        return 1.0
    height, width = foreground_mask.shape
    x1 = max(0, min(width, int(math.floor(box[0]))))
    y1 = max(0, min(height, int(math.floor(box[1]))))
    x2 = max(0, min(width, int(math.ceil(box[2]))))
    y2 = max(0, min(height, int(math.ceil(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(foreground_mask[y1:y2, x1:x2].mean())


def filter_by_foreground(
    candidates: list[Candidate],
    foreground_mask: np.ndarray | None,
    min_fraction: float,
) -> list[Candidate]:
    if foreground_mask is None:
        return candidates
    return [
        candidate
        for candidate in candidates
        if box_foreground_fraction(foreground_mask, candidate.box_xyxy) >= min_fraction
    ]


def hand_candidate_is_detail(candidate: Candidate | None) -> bool:
    if candidate is None:
        return False
    prompt = candidate.prompt.lower()
    return (
        candidate.source in {"dino_partial_hand", "dino_visible_partial_hand"}
        or (
            any(token in prompt for token in HAND_DETAIL_PROMPT_TOKENS)
            and not any(token in prompt for token in EXPLICIT_WHOLE_HAND_PROMPT_TOKENS)
        )
    )


def hand_candidate_is_whole(candidate: Candidate | None) -> bool:
    if candidate is None or hand_candidate_is_detail(candidate):
        return False
    prompt = candidate.prompt.lower().strip()
    return (
        candidate.source
        in {
            "dino_whole_hand",
            "dino_hand_union",
            "dino_visible_hand",
            "dino_visible_hand_union",
        }
        or prompt == "hand"
        or any(token in prompt for token in WHOLE_HAND_PROMPT_TOKENS)
    )


def union_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    expansion: float,
    min_margin: float = 0.0,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    margin_x = max(min_margin * width, expansion * (x2 - x1))
    margin_y = max(min_margin * height, expansion * (y2 - y1))
    return (
        max(0.0, x1 - margin_x),
        max(0.0, y1 - margin_y),
        min(float(width), x2 + margin_x),
        min(float(height), y2 + margin_y),
    )


def compact_hand_candidates(
    hands: list[Candidate],
    product: Candidate | None,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[Candidate]:
    """Prefer a whole-hand proposal and use palm/finger boxes only as support."""
    if not hands:
        return []

    whole_hands = [candidate for candidate in hands if hand_candidate_is_whole(candidate)]
    details = [candidate for candidate in hands if hand_candidate_is_detail(candidate)]
    generic_hands = [
        candidate for candidate in hands if candidate not in whole_hands and candidate not in details
    ]
    primary_pool = whole_hands or generic_hands or details

    def seed_rank(candidate: Candidate) -> float:
        relation = relation_bonus(
            candidate,
            [product] if product is not None else [],
            width,
            height,
        )
        completeness = 0.18 if hand_candidate_is_whole(candidate) else 0.0
        return candidate.score + relation + completeness

    seed = max(primary_pool, key=seed_rank)
    if product is not None:
        product_details = [
            candidate
            for candidate in details
            if box_gap_norm(candidate.box_xyxy, product.box_xyxy, width, height)
            <= args.compact_hand_max_product_gap
        ]
    else:
        product_details = details

    nearby_details = [
        candidate
        for candidate in product_details
        if (
            box_center_distance_norm(candidate.box_xyxy, seed.box_xyxy, width, height)
            <= args.compact_hand_cluster_radius
            or box_gap_norm(candidate.box_xyxy, seed.box_xyxy, width, height)
            <= args.compact_hand_max_product_gap
        )
        and candidate.score >= seed.score - args.compact_hand_score_window
    ]
    merged_box = union_boxes([seed.box_xyxy, *(item.box_xyxy for item in nearby_details)])
    visible_hand_mode = getattr(args, "visible_hand_mode", False)
    if visible_hand_mode:
        # A partial observation is a valid hand annotation. Never expand a
        # finger/palm proposal toward an imagined, occluded wrist or palm.
        final_box = merged_box
        if hand_candidate_is_whole(seed):
            source = "dino_visible_hand_union" if nearby_details else "dino_visible_hand"
            prompt = "visible human hand pixels from wrist to fingertips"
        else:
            source = "dino_visible_partial_hand"
            prompt = "visible fingers or palm of human hand"
    elif hand_candidate_is_whole(seed):
        final_box = merged_box
        source = "dino_hand_union" if nearby_details else "dino_whole_hand"
        prompt = "complete human hand from wrist to fingertips"
    else:
        final_box = expand_box(
            merged_box,
            width,
            height,
            args.compact_hand_expansion,
            args.compact_hand_min_margin,
        )
        source = "dino_partial_hand"
        prompt = "partial palm and fingers"

    if box_area(final_box) <= 0:
        return hands
    return [
        Candidate(
            HAND_CLASS_ID,
            "hand",
            final_box,
            max([seed.score, *(item.score for item in nearby_details)]),
            prompt,
            source=source,
        )
    ]

def box_compatibility(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> float:
    if second is None:
        return 0.0
    iou = box_iou(first, second)
    center = math.exp(-6.0 * box_center_distance_norm(first, second, width, height))
    area = box_area_similarity(first, second)
    return 0.65 * iou + 0.25 * center + 0.10 * area


def overlap_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return intersection / max(1e-9, min(box_area(first), box_area(second)))


def competitor_for_frame(frame: FrameDetections, role: str) -> Candidate | None:
    return frame.best_shoe if role == "shoe" else frame.best_arm


def temporal_competitor_support(
    frames: list[FrameDetections],
    index: int,
    role: str,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> int:
    current = competitor_for_frame(frames[index], role)
    if current is None:
        return 0
    support = 0
    radius = max(0, args.competitor_temporal_radius)
    for neighbor_index in range(max(0, index - radius), min(len(frames), index + radius + 1)):
        if neighbor_index == index:
            continue
        neighbor = competitor_for_frame(frames[neighbor_index], role)
        if neighbor is not None and box_compatibility(
            current.box_xyxy, neighbor.box_xyxy, width, height
        ) >= args.competitor_temporal_compatibility:
            support += 1
    return support


def competitor_guard_metrics(
    hand: TrackResult | None,
    frame: FrameDetections,
    competitor: Candidate | None,
    temporal_support: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    if hand is None or competitor is None:
        return {
            "candidate": asdict(competitor) if competitor is not None else None,
            "overlap_over_smaller": 0.0,
            "box_iou": 0.0,
            "hand_dino_score": frame.best_hand.score if frame.best_hand is not None else 0.0,
            "score_gap": 0.0,
            "temporal_support": temporal_support,
            "would_reject": False,
        }
    hand_score = frame.best_hand.score if frame.best_hand is not None else hand.output_score
    overlap = overlap_over_smaller(hand.obj.box_xyxy, competitor.box_xyxy)
    score_gap = competitor.score - hand_score
    would_reject = (
        competitor.score >= args.competitor_min_score
        and score_gap >= args.competitor_score_margin
        and overlap >= args.competitor_min_overlap
        and temporal_support >= args.competitor_min_temporal_support
    )
    return {
        "candidate": asdict(competitor),
        "overlap_over_smaller": overlap,
        "box_iou": box_iou(hand.obj.box_xyxy, competitor.box_xyxy),
        "hand_dino_score": hand_score,
        "score_gap": score_gap,
        "temporal_support": temporal_support,
        "would_reject": would_reject,
    }


def _cand(d):
    """jsonl 한 항목 -> Candidate. box 는 튜플로 되돌린다."""
    if not d:
        return None
    return Candidate(
        class_id=int(d["class_id"]), class_name=d["class_name"],
        box_xyxy=tuple(float(x) for x in d["box_xyxy"]),
        score=float(d["score"]), prompt=d.get("prompt", ""),
        source=d.get("source", "dino"),
    )


def save_product_cache(path: Path, tracks: dict[int, "TrackResult"]) -> None:
    """제품 추적 결과를 저장한다.

    2패스에서 2차가 바꾸는 것은 손 시드뿐이고 제품 추적은 손과 무관하게 같은 결과를 낸다
    (실측: 1,595프레임 중 1,594개 동일). 그런데도 2차가 제품 전파를 다시 도는 것은 낭비이고,
    실측으로 전체의 25% 를 차지한다.

    마스크는 프레임당 640x480 bool 이라 그대로 두면 무겁다. packbits 로 8배 줄이고
    하나의 평면 배열에 이어붙여 offset 으로 자른다(멤버 1,595개짜리 zip 을 피한다).
    """
    order = sorted(tracks)
    meta: list[dict] = []
    packed: list[np.ndarray] = []
    shape = (0, 0)
    for index in order:
        track = tracks[index]
        obj = track.obj
        has_mask = track.mask is not None
        if has_mask:
            pixels = np.asarray(track.mask, dtype=bool)
            shape = pixels.shape
            packed.append(np.packbits(pixels.reshape(-1)))
        else:
            packed.append(np.zeros(0, dtype=np.uint8))
        meta.append(
            {
                "i": int(index),
                "class_id": int(obj.class_id),
                "class_name": obj.class_name,
                "box": [float(x) for x in obj.box_xyxy],
                "score": float(obj.score),
                "prompt": obj.prompt,
                "mask_iou": float(obj.mask_iou),
                "mask_score": float(obj.mask_score),
                "mask_area_ratio": float(obj.mask_area_ratio),
                "mask_component_count": int(obj.mask_component_count),
                "mask_valid": bool(obj.mask_valid),
                "polygon": (
                    [[float(x), float(y)] for x, y in obj.polygon_points]
                    if obj.polygon_points
                    else None
                ),
                "source": track.source,
                "output_score": float(track.output_score),
                "has_mask": has_mask,
            }
        )
    flat = np.concatenate(packed) if packed else np.zeros(0, dtype=np.uint8)
    offsets = np.cumsum([0] + [len(p) for p in packed]).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        shape=np.array(shape, dtype=np.int64),
        flat=flat,
        offsets=offsets,
    )
    print(f"[cache] product tracks saved frames={len(order)} path={path}")


def load_product_cache(path: str) -> dict[int, "TrackResult"]:
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    height, width = (int(x) for x in blob["shape"])
    flat = blob["flat"]
    offsets = blob["offsets"]
    tracks: dict[int, TrackResult] = {}
    for slot, item in enumerate(meta):
        mask = None
        if item["has_mask"]:
            chunk = flat[offsets[slot] : offsets[slot + 1]]
            mask = (
                np.unpackbits(chunk)[: height * width]
                .reshape(height, width)
                .astype(bool)
            )
        obj = FinalObject(
            item["class_id"],
            item["class_name"],
            tuple(item["box"]),
            item["score"],
            item["prompt"],
            item["mask_iou"],
            item["mask_score"],
            item["mask_area_ratio"],
            item["mask_component_count"],
            item["mask_valid"],
            [tuple(p) for p in item["polygon"]] if item["polygon"] else None,
        )
        tracks[int(item["i"])] = TrackResult(
            obj, mask, item["source"], item["output_score"]
        )
    print(f"[cache] product tracks loaded frames={len(tracks)} path={path}")
    return tracks


def load_dino_cache(path: str, files: list[Path], args: argparse.Namespace):
    """1차 run 의 detections.jsonl 로 DINO 스캔을 대체한다.

    DINO 는 결정적이라 같은 입력이면 결과가 같다. 2패스에서 2차가 스캔을 다시 도는 것은
    순수 낭비인데, 실측으로 전체의 30~36% 를 차지한다.

    jsonl 에는 fallback_hand / fallback_object 가 없다. 이 둘은 select_final_candidates
    의 결과이고 previous_object_box 라는 프레임 간 상태에 의존하므로, 저장된 best_object
    로 그 상태를 그대로 재생해 다시 계산한다(갱신 규칙이 단순해 재현이 가능하다).
    fallback 은 SAM3 가 추적 못 한 프레임에서 실제로 쓰이므로 생략할 수 없다.
    """
    byname = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            d = json.loads(line)
            byname[Path(d["image"]).stem] = d
    frames: list[FrameDetections] = []
    previous_object_box = None
    previous_object_age = 0
    missing = 0
    for image_path in files:
        d = byname.get(image_path.stem)
        if d is None:
            missing += 1
            frames.append(FrameDetections([], [], None, None, None, None))
            continue
        hands = [c for c in (_cand(x) for x in d.get("hand_candidates", [])) if c]
        objects = [c for c in (_cand(x) for x in d.get("object_candidates", [])) if c]
        best_hand = _cand(d.get("best_hand"))
        best_object = _cand(d.get("best_object"))
        with Image.open(image_path) as im:
            width, height = im.size
        selected = select_final_candidates(
            hands + objects, width, height, args,
            previous_object_box=previous_object_box,
            previous_object_age=previous_object_age,
        )
        fallback_object = next((i for i in selected if i.class_id != HAND_CLASS_ID), None)
        fallback_hand = next((i for i in selected if i.class_id == HAND_CLASS_ID), None)
        frames.append(FrameDetections(
            hand_candidates=hands, object_candidates=objects,
            best_hand=best_hand, best_object=best_object,
            fallback_hand=fallback_hand, fallback_object=fallback_object,
            shoe_candidates=[c for c in (_cand(x) for x in d.get("shoe_candidates", [])) if c],
            arm_candidates=[c for c in (_cand(x) for x in d.get("arm_candidates", [])) if c],
        ))
        if best_object is not None:
            previous_object_box = best_object.box_xyxy
            previous_object_age = 0
        elif previous_object_box is not None:
            previous_object_age += 1
            if previous_object_age > args.temporal_max_gap:
                previous_object_box = None
                previous_object_age = 0
    print(f"[dino-cache] {len(files)}프레임 중 {len(files)-missing}개 복원, 없음 {missing}",
          flush=True)
    return frames


def dino_detect_all(
    files: list[Path],
    dino_processor,
    dino_model,
    args: argparse.Namespace,
) -> list[dict]:
    """DINO 검출만 따로, 배치로 + CPU 준비를 미리 해가며 돈다.

    한 장씩 돌면 CPU(열기·리사이즈·후처리)와 GPU(forward)가 번갈아 돌아서 GPU 가 20~40%
    를 논다. 여기서는 준비 스레드가 다음 배치를 미리 만들어 두므로 GPU 는 계속 돈다.

    프레임 간 상태에 의존하는 선택 로직은 전혀 건드리지 않는다. 이 함수는 프레임별로
    독립적인 것(원본 후보, 전경 마스크)만 만들고, 순서가 중요한 계산은 호출부가 한다.
    """
    batch = max(1, int(getattr(args, "dino_batch", 1)))
    # (텍스트, 클래스, 이름, box 임계, text 임계, 후보 상한, 저장 키)
    texts = [
        (grounding_text(args.hand_prompts), HAND_CLASS_ID, "hand",
         args.hand_box_threshold, args.text_threshold, args.max_hand_candidates, "hands"),
    ]
    if not getattr(args, "hand_only", False):
        texts.append((grounding_text(args.object_prompts), args.product_class_id, args.product_name,
                      args.object_box_threshold, args.text_threshold,
                      args.max_object_candidates, "objects"))
    if args.shoe_competitor_mode != "off":
        texts.append((grounding_text(args.shoe_competitor_prompts),
                      SHOE_COMPETITOR_CLASS_ID, "shoe_competitor",
                      args.competitor_box_threshold, args.competitor_text_threshold,
                      args.max_competitor_candidates, "shoes"))
    if args.arm_competitor_mode != "off":
        texts.append((grounding_text(args.arm_competitor_prompts),
                      ARM_COMPETITOR_CLASS_ID, "arm_competitor",
                      args.competitor_box_threshold, args.competitor_text_threshold,
                      args.max_competitor_candidates, "arms"))

    chunks = [files[i : i + batch] for i in range(0, len(files), batch)]

    def prepare(chunk: list[Path]) -> dict:
        """CPU 쪽 전부. 준비 스레드에서 돈다."""
        images = [Image.open(p).convert("RGB") for p in chunk]
        masks = [
            foreground_mask_for_image(
                im, args.background_references, args.background_pixel_threshold
            )
            for im in images
        ]
        inputs = {
            key: dino_processor(images=images, text=[text] * len(images), return_tensors="pt")
            for text, _, _, _, _, _, key in texts
        }
        return {"images": images, "masks": masks, "inputs": inputs, "count": len(chunk)}

    out: list[dict] = []
    done = 0
    # 준비 스레드는 2개면 충분하다. GPU 한 배치 도는 동안 다음 배치가 준비되기만 하면 된다.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending: deque = deque()
        for chunk in chunks[:3]:
            pending.append(pool.submit(prepare, chunk))
        next_submit = 3
        with torch.inference_mode():
            while pending:
                ready = pending.popleft().result()
                if next_submit < len(chunks):
                    pending.append(pool.submit(prepare, chunks[next_submit]))
                    next_submit += 1
                per_frame = [
                    {"width": im.width, "height": im.height,
                     "foreground_mask": mask, "background_reference": ref}
                    for im, (mask, ref) in zip(ready["images"], ready["masks"])
                ]
                for text, class_id, class_name, box_thr, text_thr, max_c, key in texts:
                    inputs = ready["inputs"][key].to(args.dino_device)
                    with dino_autocast(args.dino_device):
                        outputs = dino_model(**inputs)
                    results = dino_processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=box_thr,
                        text_threshold=text_thr,
                        target_sizes=[(im.height, im.width) for im in ready["images"]],
                    )
                    for slot, result in enumerate(results):
                        per_frame[slot][key] = candidates_from_results(
                            result, class_id, class_name, text,
                            ready["images"][slot].width, ready["images"][slot].height, max_c,
                        )
                for slot in per_frame:
                    for key in ("hands", "objects", "shoes", "arms"):
                        slot.setdefault(key, [])
                out.extend(per_frame)
                done += ready["count"]
                if args.verbose or done >= len(files) or done % 200 < batch:
                    print(f"[dino {done}/{len(files)}] batch={batch}", flush=True)
    return out


def scan_with_dino(
    files: list[Path],
    dino_processor,
    dino_model,
    args: argparse.Namespace,
) -> list[FrameDetections]:
    frames: list[FrameDetections] = []
    previous_object_box: tuple[float, float, float, float] | None = None
    previous_object_age = 0
    detected = dino_detect_all(files, dino_processor, dino_model, args)

    for index, image_path in enumerate(files):
        raw = detected[index]
        width, height = raw["width"], raw["height"]
        foreground_mask = raw["foreground_mask"]
        background_reference = raw["background_reference"]
        hands = raw["hands"]
        objects = raw["objects"]
        shoes = raw["shoes"]
        arms = raw["arms"]
        hands = filtered_candidates(hands, width, height, args)
        objects = filtered_candidates(objects, width, height, args)
        shoes = nms(shoes, args.nms_iou)
        arms = nms(arms, args.nms_iou)
        hands = filter_by_foreground(
            hands, foreground_mask, args.background_min_hand_foreground
        )
        objects = filter_by_foreground(
            objects, foreground_mask, args.background_min_object_foreground
        )

        best_object = (
            max(
                objects,
                key=lambda candidate: object_selection_score(
                    candidate,
                    hands,
                    previous_object_box,
                    previous_object_age,
                    width,
                    height,
                    args,
                ),
            )
            if objects
            else None
        )
        hands = compact_hand_candidates(hands, best_object, width, height, args)
        distinct_hands = [
            candidate
            for candidate in hands
            if best_object is None
            or not cross_class_duplicate(
                candidate,
                best_object,
                args.cross_class_duplicate_iou,
                args.cross_class_duplicate_area_similarity,
            )
        ]
        best_hand = (
            max(
                distinct_hands,
                key=lambda candidate: candidate.score
                + relation_bonus(
                    candidate,
                    [best_object] if best_object is not None else objects,
                    width,
                    height,
                ),
            )
            if distinct_hands
            else None
        )

        selected = select_final_candidates(
            hands + objects,
            width,
            height,
            args,
            previous_object_box=previous_object_box,
            previous_object_age=previous_object_age,
        )
        fallback_object = next((item for item in selected if item.class_id != HAND_CLASS_ID), None)
        fallback_hand = next((item for item in selected if item.class_id == HAND_CLASS_ID), None)

        frames.append(
            FrameDetections(
                hand_candidates=hands,
                object_candidates=objects,
                best_hand=best_hand,
                best_object=best_object,
                fallback_hand=fallback_hand,
                fallback_object=fallback_object,
                shoe_candidates=shoes,
                arm_candidates=arms,
                best_shoe=max(shoes, key=lambda candidate: candidate.score) if shoes else None,
                best_arm=max(arms, key=lambda candidate: candidate.score) if arms else None,
                foreground_mask=foreground_mask,
                background_reference=background_reference,
            )
        )
        if best_object is not None:
            previous_object_box = best_object.box_xyxy
            previous_object_age = 0
        elif previous_object_box is not None:
            previous_object_age += 1
            if previous_object_age > args.temporal_max_gap:
                previous_object_box = None
                previous_object_age = 0

        if args.verbose or index == 0 or (index + 1) % 100 == 0:
            print(
                f"[dino {index + 1}/{len(files)}] {image_path.name} "
                f"hand={len(hands)} object={len(objects)} "
                f"shoe={len(shoes)} arm={len(arms)}"
            )
    return frames


def anchor_support(
    frames: list[FrameDetections],
    index: int,
    role: str,
    width: int,
    height: int,
    radius: int,
) -> float:
    candidate = frames[index].best_object if role == "object" else frames[index].best_hand
    if candidate is None:
        return 0.0
    support = 0.0
    for neighbor_index in range(max(0, index - radius), min(len(frames), index + radius + 1)):
        if neighbor_index == index:
            continue
        neighbor = (
            frames[neighbor_index].best_object
            if role == "object"
            else frames[neighbor_index].best_hand
        )
        if neighbor is not None:
            support = max(
                support,
                box_compatibility(candidate.box_xyxy, neighbor.box_xyxy, width, height),
            )
    return support


def select_anchor(
    files: list[Path],
    frames: list[FrameDetections],
    role: str,
    args: argparse.Namespace,
    start: int = 0,
    end: int | None = None,
    keep=None,
    area_cap: float | None = None,
    whole_bonus: bool = True,
    fill: dict | None = None,
    fill_min: float = 0.0,
) -> tuple[int, Candidate, float] | None:
    """start~end 구간 안에서만 시드를 고른다. 기본값은 영상 전체(기존 동작).

    keep 를 주면 그 조건을 만족하는 프레임만 후보로 본다(겹침/비겹침 분리용).
    """
    min_score = (
        args.video_product_seed_threshold
        if role == "object"
        else args.video_hand_seed_threshold
    )
    ranked: list[tuple[float, int, Candidate]] = []
    last = len(frames) - 1 if end is None else min(end, len(frames) - 1)
    for index, frame in enumerate(frames):
        if not (start <= index <= last):
            continue
        if keep is not None and not keep(frame):
            continue
        candidate = frame.best_object if role == "object" else frame.best_hand
        if candidate is None or candidate.score < min_score:
            continue
        # 채움비(마스크 면적 / DINO 박스 면적) 하한. 팔까지 물린 박스는 안이 비어
        # 채움비가 낮다. 0.45 이상이면 97~99% 가 정답의 1.3 배 이내였다.
        if fill_min > 0 and fill is not None:
            if float(fill.get(files[index].stem, 0.0)) < fill_min:
                continue
        # 크기 제약은 candidate 을 얻은 뒤에 본다. 이 검사를 할당보다 앞에 두어
        # UnboundLocalError 가 났고, 그 탓에 실험 17(시드 크기 제약)이 무효가 됐다.
        if area_cap is not None and box_area(candidate.box_xyxy) > area_cap:
            continue
        with Image.open(files[index]) as image:
            width, height = image.size
        support = anchor_support(
            frames,
            index,
            role,
            width,
            height,
            args.video_seed_support_radius,
        )
        relation = relation_bonus(
            candidate,
            frame.hand_candidates if role == "object" else frame.object_candidates,
            width,
            height,
        )
        # 완전성 보너스는 "손 전체가 보이는" 후보를 우대하는데, 팔까지 물린 박스가
        # 바로 그 '전체' 로 판정된다. 제품에 가려진 구간에서는 손이 쪼개져 있는 것이
        # 정상이므로 이 가점을 끈다.
        completeness = (
            0.12 if role == "hand" and whole_bonus
            and hand_candidate_is_whole(candidate) else 0.0
        )
        rank = candidate.score + 0.15 * support + 0.08 * relation + completeness
        ranked.append((rank, index, candidate))
    if not ranked:
        return None
    rank, index, candidate = max(ranked, key=lambda item: item[0])
    return index, candidate, rank


@contextmanager
def ordered_video_directory(files: list[Path]) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="sam3_video_frames.") as temp_dir:
        root = Path(temp_dir)
        for index, source in enumerate(files):
            suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_EXTS else ".jpg"
            os.symlink(source.resolve(), root / f"{index:08d}{suffix}")
        yield root


def normalized_xywh(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, (x2 - x1) / width)),
        max(0.0, min(1.0, (y2 - y1) / height)),
    ]



def adaptive_hand_support(
    dino_candidate: Candidate | None,
    previous_box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[tuple[float, float, float, float] | None, float]:
    support_box = dino_candidate.box_xyxy if dino_candidate is not None else previous_box
    expansion = args.hand_mask_support_expansion
    if support_box is None:
        return None, expansion

    if getattr(args, "visible_hand_mode", False):
        # The previous frame can contain substantially more of the hand than
        # the current frame. Using it as a spatial union would re-introduce
        # occluded palm/wrist pixels or forearm, so only the current observation
        # constrains SAM3 when one exists.
        return support_box, min(
            max(0.0, args.visible_hand_support_expansion),
            args.hand_max_mask_support_expansion,
        )

    if hand_candidate_is_detail(dino_candidate):
        expansion = max(expansion, args.hand_partial_mask_support_expansion)
        if previous_box is not None:
            related = (
                box_iou(support_box, previous_box) > 0.0
                or box_gap_norm(support_box, previous_box, width, height)
                <= args.compact_hand_cluster_radius
            )
            if related:
                seed_area = max(1.0, box_area(support_box))
                previous_area = max(1.0, box_area(previous_box))
                size_expansion = max(0.0, (math.sqrt(previous_area / seed_area) - 1.0) * 0.5)
                expansion = max(expansion, size_expansion)
                support_box = union_boxes([support_box, previous_box])
    elif previous_box is not None and box_compatibility(
        support_box, previous_box, width, height
    ) >= args.min_temporal_compatibility:
        support_box = union_boxes([support_box, previous_box])

    return support_box, min(expansion, args.hand_max_mask_support_expansion)


def component_box(stats: np.ndarray, label: int) -> tuple[float, float, float, float]:
    x, y, component_width, component_height, _ = stats[label]
    return (
        float(x),
        float(y),
        float(x + component_width),
        float(y + component_height),
    )


def constrain_hand_mask(
    mask: np.ndarray,
    support_box: tuple[float, float, float, float] | None,
    previous_box: tuple[float, float, float, float] | None,
    product_box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    expansion: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    """Keep related visible hand components without filling product-occluded gaps."""
    clipped = np.asarray(mask, dtype=bool)
    if support_box is not None:
        x1, y1, x2, y2 = expand_box(support_box, width, height, expansion)
        left = max(0, int(math.floor(x1)))
        top = max(0, int(math.floor(y1)))
        right = min(width, int(math.ceil(x2)))
        bottom = min(height, int(math.ceil(y2)))
        allowed = np.zeros((height, width), dtype=bool)
        allowed[top:bottom, left:right] = True
        clipped = np.logical_and(clipped, allowed)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        clipped.astype(np.uint8), connectivity=8
    )
    foreground_labels = list(range(1, component_count))
    if not foreground_labels:
        return np.zeros_like(clipped, dtype=bool), 0
    if len(foreground_labels) == 1:
        return labels == foreground_labels[0], 1

    primary_label = max(
        foreground_labels,
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
    )
    primary_area = max(1, int(stats[primary_label, cv2.CC_STAT_AREA]))
    primary_box = component_box(stats, primary_label)
    kept_labels = [primary_label]

    for label in foreground_labels:
        if label == primary_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(4, int(primary_area * args.hand_component_min_area_ratio)):
            continue
        box = component_box(stats, label)
        near_primary = (
            box_gap_norm(box, primary_box, width, height) <= args.hand_component_max_gap
        )
        near_product = (
            product_box is not None
            and box_gap_norm(box, product_box, width, height)
            <= args.hand_component_product_gap
        )
        temporal_support = (
            previous_box is not None
            and (
                box_iou(box, previous_box) > 0.0
                or box_gap_norm(box, previous_box, width, height)
                <= args.hand_component_max_gap
            )
        )
        if near_primary or near_product or temporal_support:
            kept_labels.append(label)

    grouped = np.isin(labels, np.asarray(kept_labels, dtype=labels.dtype))
    return grouped.astype(bool), len(kept_labels)

def output_candidates(
    outputs: dict,
    class_id: int,
    class_name: str,
    prompt: str,
    dino_candidate: Candidate | None,
    product_candidate: Candidate | None,
    previous_box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[tuple[float, TrackResult]]:
    masks = np.asarray(outputs.get("out_binary_masks", []))
    probs = np.asarray(outputs.get("out_probs", np.ones(len(masks), dtype=np.float32)))
    # [계측] SAM3 가 프레임마다 후보 마스크를 몇 개나 주는지, 확률이 어떻게 갈리는지.
    # 후보가 여러 개면 "다른 것을 고르는" 여지가 있고, 하나뿐이면 선택 규칙은 무의미하다.
    if os.getenv("PROBE_MASKS"):
        head = " ".join(f"{float(x):.3f}" for x in probs[:6])
        print(f"[probe] role={class_name} masks={len(masks)} probs=[{head}]", flush=True)
    ranked: list[tuple[float, TrackResult]] = []
    for mask, probability in zip(masks, probs):
        score = float(probability)
        if score < args.video_min_output_score:
            continue
        mask = np.asarray(mask, dtype=bool)
        mask_component_count = 1
        if class_id == HAND_CLASS_ID:
            support_box, support_expansion = adaptive_hand_support(
                dino_candidate,
                previous_box,
                width,
                height,
                args,
            )
            mask, mask_component_count = constrain_hand_mask(
                mask,
                support_box,
                previous_box,
                product_candidate.box_xyxy if product_candidate is not None else None,
                width,
                height,
                support_expansion,
                args,
            )
        box = mask_to_box(mask)
        if box is None:
            continue
        candidate = Candidate(class_id, class_name, box, score, prompt, source="sam3_video")
        if not role_shape_ok(candidate, width, height, args):
            continue
        dino_compatibility = (
            box_compatibility(box, dino_candidate.box_xyxy, width, height)
            if dino_candidate is not None
            else 0.0
        )
        previous_compatibility = box_compatibility(box, previous_box, width, height)
        rank = score + 0.35 * dino_compatibility + 0.25 * previous_compatibility
        polygon = mask_to_polygon(
            mask,
            width,
            height,
            args.polygon_epsilon,
            args.min_polygon_points,
        )
        prompt_iou = (
            box_iou(box, dino_candidate.box_xyxy) if dino_candidate is not None else 0.0
        )
        prompt_area = (
            box_area(dino_candidate.box_xyxy) if dino_candidate is not None else box_area(box)
        )
        final = FinalObject(
            class_id,
            class_name,
            box,
            score,
            prompt,
            mask_iou=prompt_iou,
            mask_score=score,
            mask_area_ratio=box_area(box) / max(1e-9, prompt_area),
            mask_component_count=mask_component_count,
            mask_valid=polygon is not None,
            polygon_points=polygon,
        )
        ranked.append((rank, TrackResult(final, mask, "sam3_video", score)))
    return ranked


def select_track_output(
    outputs: dict,
    role: str,
    frame_detection: FrameDetections,
    previous_box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> TrackResult | None:
    class_id = HAND_CLASS_ID if role == "hand" else args.product_class_id
    class_name = "hand" if role == "hand" else args.product_name
    prompt = args.hand_prompts[0] if role == "hand" else args.object_prompts[0]
    dino_candidate = (
        frame_detection.best_hand if role == "hand" else frame_detection.best_object
    )
    ranked = output_candidates(
        outputs,
        class_id,
        class_name,
        prompt,
        dino_candidate,
        frame_detection.best_object,
        previous_box,
        width,
        height,
        args,
    )
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


_FILL_CACHE = {}


def load_fill(path: str) -> dict:
    """1차 run 에서 뽑은 프레임별 채움비. 2패스 시드 선정에 쓴다."""
    if not path:
        return {}
    if path not in _FILL_CACHE:
        try:
            _FILL_CACHE[path] = json.loads(Path(path).read_text())
        except Exception as exc:
            print(f"[warn] 채움비 파일을 못 읽음: {exc}", flush=True)
            _FILL_CACHE[path] = {}
    return _FILL_CACHE[path]


def _best_obj_id(outputs):
    """시드 출력에서 확률이 가장 높은 객체의 id.

    add_prompt 의 첫 반환값은 frame_idx 이지 객체 id 가 아니다(그렇게 착각해 프레임
    번호를 obj_id 로 넘기면 기존 객체를 다듬는 대신 새 객체가 생긴다). 객체 id 는
    출력 dict 의 out_obj_ids 에 들어 있다.
    """
    ids = outputs.get("out_obj_ids") if isinstance(outputs, dict) else None
    if ids is None:
        return None
    ids = np.asarray(ids).reshape(-1)
    if ids.size == 0:
        return None
    probs = np.asarray(outputs.get("out_probs", [])).reshape(-1)
    if probs.size == ids.size:
        return int(ids[int(np.argmax(probs))])
    return int(ids[0])


def hand_refine_points(hand_box, product_candidate, width, height, args):
    """손 시드를 다듬을 점들. 좌표는 0~1 상대값.

    양성 — 손 박스 중심과 그 아래쪽(손바닥·손가락은 아래에 있다).
           위쪽은 팔일 수 있으므로 양성 점을 두지 않는다.
    음성 — 제품 박스 중심(제품은 손이 아니다), 그리고 손 박스 위쪽 바깥(팔 방향).
    """
    x1, y1, x2, y2 = hand_box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pts, labs = [(cx, cy), (cx, y2 - h * 0.25)], [1, 1]

    if product_candidate is not None:
        px1, py1, px2, py2 = product_candidate.box_xyxy
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        # 제품 중심이 손 박스 안에 있을 때만 쓴다. 밖이면 굳이 알려줄 필요가 없고,
        # 잘못 찍으면 손의 일부를 부정하게 된다.
        if x1 <= pcx <= x2 and y1 <= pcy <= y2:
            pts.append((pcx, pcy))
            labs.append(0)

    if getattr(args, "hand_point_arm_negative", True):
        arm_y = y1 - h * float(getattr(args, "hand_point_arm_offset", 0.35))
        if arm_y > 2:
            pts.append((cx, arm_y))
            labs.append(0)

    rel = [(min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))
           for x, y in pts]
    return rel, labs


def track_role_pass(
    model,
    state: dict,
    files: list[Path],
    frames: list[FrameDetections],
    anchor: tuple[int, Candidate, float],
    role: str,
    role_text: str,
    args: argparse.Namespace,
    start_index: int = 0,
    end_index: int | None = None,
    source: str = "sam3_video",
) -> dict[int, TrackResult]:
    anchor_index, anchor_candidate, anchor_rank = anchor
    end_index = len(files) - 1 if end_index is None else min(end_index, len(files) - 1)
    start_index = max(0, start_index)
    if not (start_index <= anchor_index <= end_index):
        return {}

    with Image.open(files[anchor_index]) as image:
        width, height = image.size
    print(
        f"[video] {role} seed frame={anchor_index} file={files[anchor_index].name} "
        f"dino={anchor_candidate.score:.3f} rank={anchor_rank:.3f} "
        f"range={start_index}:{end_index} source={source}"
    )
    with autocast_for(args.sam3_device, args.sam3_autocast_dtype):
        _, seed_outputs = model.add_prompt(
            state,
            frame_idx=anchor_index,
            text_str=role_text,
            boxes_xywh=[normalized_xywh(anchor_candidate.box_xyxy, width, height)],
            box_labels=[1],
        )
        # 점 프롬프트로 시드를 교정한다. SAM3 는 점을 텍스트·박스와 동시에 못 받으므로
        # 위에서 객체를 만든 뒤 obj_id 를 지정해 두 번째 호출로 다듬는다.
        #
        # 왜: 지금까지 SAM3 에 "이 박스 안에 손이 있다" 만 알려줬다. 그래서 팔로 번지는
        # 것도, 제품에 가려 사라지는 것도 알려줄 방법이 없었다. 음성 점이면 "제품은 손이
        # 아니다", "위쪽 팔은 손이 아니다" 를 직접 말할 수 있다.
        if role == "hand" and getattr(args, "hand_point_refine", False):
            pts, labs = hand_refine_points(
                anchor_candidate.box_xyxy,
                frames[anchor_index].best_object,
                width,
                height,
                args,
            )
            oid = _best_obj_id(seed_outputs)
            if pts and oid is not None:
                try:
                    _, seed_outputs = model.add_prompt(
                        state,
                        frame_idx=anchor_index,
                        points=pts,
                        point_labels=labs,
                        obj_id=oid,
                        rel_coordinates=True,
                    )
                    print(f"[video] hand point refine frame={anchor_index} "
                          f"obj={oid} pos={labs.count(1)} neg={labs.count(0)}", flush=True)
                except Exception as exc:      # 점 교정이 실패해도 박스 시드로 계속 간다
                    print(f"[warn] point refine skipped: {type(exc).__name__}: {exc}",
                          flush=True)
    seed = select_track_output(
        seed_outputs,
        role,
        frames[anchor_index],
        anchor_candidate.box_xyxy,
        width,
        height,
        args,
    )
    results: dict[int, TrackResult] = {}
    if seed is not None:
        seed.source = source
        results[anchor_index] = seed

    previous_box = seed.obj.box_xyxy if seed is not None else anchor_candidate.box_xyxy
    forward_count = max(1, end_index - anchor_index + 1)
    with autocast_for(args.sam3_device, args.sam3_autocast_dtype):
        for frame_index, outputs in model.propagate_in_video(
            state,
            start_frame_idx=anchor_index,
            max_frame_num_to_track=forward_count,
            reverse=False,
        ):
            if not (anchor_index <= frame_index <= end_index):
                continue
            with Image.open(files[frame_index]) as image:
                frame_width, frame_height = image.size
            selected = select_track_output(
                outputs,
                role,
                frames[frame_index],
                previous_box,
                frame_width,
                frame_height,
                args,
            )
            if selected is not None:
                selected.source = source
                results[frame_index] = selected
                previous_box = selected.obj.box_xyxy

    previous_box = seed.obj.box_xyxy if seed is not None else anchor_candidate.box_xyxy
    reverse_count = max(1, anchor_index - start_index + 1)
    with autocast_for(args.sam3_device, args.sam3_autocast_dtype):
        for frame_index, outputs in model.propagate_in_video(
            state,
            start_frame_idx=anchor_index,
            max_frame_num_to_track=reverse_count,
            reverse=True,
        ):
            if not (start_index <= frame_index <= anchor_index):
                continue
            with Image.open(files[frame_index]) as image:
                frame_width, frame_height = image.size
            selected = select_track_output(
                outputs,
                role,
                frames[frame_index],
                previous_box,
                frame_width,
                frame_height,
                args,
            )
            if selected is not None:
                selected.source = source
                results[frame_index] = selected
                previous_box = selected.obj.box_xyxy
    return results


def hand_collapse_indices(
    results: dict[int, TrackResult],
    frames: list[FrameDetections],
    args: argparse.Namespace,
) -> list[int]:
    reliable_areas: list[float] = []
    reliable_widths: list[float] = []
    reliable_heights: list[float] = []
    reliable_components: list[int] = []
    collapsed: list[int] = []

    for index, frame in enumerate(frames):
        track = results.get(index)
        dino = frame.best_hand
        if track is None:
            if hand_candidate_is_whole(dino) and dino.score >= args.video_hand_seed_threshold:
                collapsed.append(index)
            continue

        x1, y1, x2, y2 = track.obj.box_xyxy
        area = max(1.0, box_area(track.obj.box_xyxy))
        track_width = max(1.0, x2 - x1)
        track_height = max(1.0, y2 - y1)
        dino_area_collapse = False
        if (
            hand_candidate_is_whole(dino)
            and dino.score >= args.video_hand_seed_threshold
        ):
            dx1, dy1, dx2, dy2 = dino.box_xyxy
            dino_area = max(1.0, box_area(dino.box_xyxy))
            dino_width = max(1.0, dx2 - dx1)
            dino_height = max(1.0, dy2 - dy1)
            dino_area_collapse = (
                area < dino_area * args.hand_collapse_area_ratio
                and (
                    track_width < dino_width * args.hand_collapse_dimension_ratio
                    or track_height < dino_height * args.hand_collapse_dimension_ratio
                )
            )

        visible_observation_supported = False
        if getattr(args, "visible_hand_mode", False) and dino is not None:
            tx1, ty1, tx2, ty2 = track.obj.box_xyxy
            dx1, dy1, dx2, dy2 = dino.box_xyxy
            intersection = max(0.0, min(tx2, dx2) - max(tx1, dx1)) * max(
                0.0, min(ty2, dy2) - max(ty1, dy1)
            )
            overlap_over_smaller = intersection / max(
                1.0, min(area, box_area(dino.box_xyxy))
            )
            visible_observation_supported = (
                overlap_over_smaller >= args.visible_hand_min_dino_overlap
                and box_area_similarity(track.obj.box_xyxy, dino.box_xyxy)
                >= args.hand_collapse_area_ratio
            )

        is_collapsed = dino_area_collapse
        if len(reliable_areas) >= args.hand_collapse_history_min:
            baseline_area = float(np.median(reliable_areas[-args.hand_collapse_history_window :]))
            baseline_width = float(np.median(reliable_widths[-args.hand_collapse_history_window :]))
            baseline_height = float(np.median(reliable_heights[-args.hand_collapse_history_window :]))
            baseline_components = float(
                np.median(reliable_components[-args.hand_collapse_history_window :])
            )
            area_small = area < baseline_area * args.hand_collapse_area_ratio
            dimension_small = (
                track_width < baseline_width * args.hand_collapse_dimension_ratio
                or track_height < baseline_height * args.hand_collapse_dimension_ratio
            )
            dino_support = (
                hand_candidate_is_whole(dino)
                and box_area(dino.box_xyxy) >= baseline_area * args.hand_recovery_min_area_ratio
                and box_area(dino.box_xyxy) >= area / max(args.hand_collapse_area_ratio, 1e-6)
            )
            component_drop = (
                baseline_components >= 1.5
                and track.obj.mask_component_count < baseline_components * 0.5
            )
            votes = sum((area_small, dimension_small, dino_support, component_drop))
            is_collapsed = (
                is_collapsed
                or votes >= 2
                or (area_small and dino_support)
            )

        if getattr(args, "visible_hand_mode", False) and (
            dino is None or visible_observation_supported
        ):
            # Size changes alone are not failures in this mode: a full hand can
            # legitimately become a few visible fingertips. Recover only when
            # current-frame DINO evidence contradicts the SAM3 observation.
            is_collapsed = False

        if is_collapsed:
            collapsed.append(index)
            continue
        reliable_areas.append(area)
        reliable_widths.append(track_width)
        reliable_heights.append(track_height)
        reliable_components.append(max(1, track.obj.mask_component_count))
    return collapsed


def grouped_indices(indices: list[int], merge_gap: int) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index - previous > merge_gap + 1:
            spans.append((start, previous))
            start = index
        previous = index
    spans.append((start, previous))
    return spans


def local_hand_reference_area(
    results: dict[int, TrackResult],
    index: int,
    excluded: set[int],
    radius: int,
) -> float:
    reliable = {
        frame_index: box_area(track.obj.box_xyxy)
        for frame_index, track in results.items()
        if frame_index not in excluded
    }
    if not reliable:
        return 0.0
    local_areas = [
        area
        for frame_index, area in reliable.items()
        if abs(frame_index - index) <= max(0, radius)
    ]
    return float(np.median(local_areas or list(reliable.values())))


def select_hand_recovery_anchor(
    frames: list[FrameDetections],
    results: dict[int, TrackResult],
    span: tuple[int, int],
    collapsed: set[int],
    used: set[int],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[int, Candidate, float] | None:
    search_start = max(0, span[0] - args.hand_recovery_search_radius)
    search_end = min(len(frames) - 1, span[1] + args.hand_recovery_search_radius)
    baseline_area = local_hand_reference_area(
        results,
        (span[0] + span[1]) // 2,
        collapsed,
        args.hand_recovery_search_radius,
    )
    ranked: list[tuple[float, int, Candidate]] = []
    for index in range(search_start, search_end + 1):
        if index in used:
            continue
        candidate = frames[index].best_hand
        if (
            not hand_candidate_is_whole(candidate)
            or candidate.score < args.video_hand_seed_threshold
        ):
            continue
        candidate_area = box_area(candidate.box_xyxy)
        if baseline_area and candidate_area < baseline_area * args.hand_recovery_min_area_ratio:
            continue
        support = anchor_support(
            frames,
            index,
            "hand",
            width,
            height,
            args.video_seed_support_radius,
        )
        outside_span_bonus = 0.08 if index < span[0] or index > span[1] else 0.0
        rank = candidate.score + 0.20 + 0.15 * support + outside_span_bonus
        ranked.append((rank, index, candidate))
    if not ranked:
        return None
    rank, index, candidate = max(ranked, key=lambda item: item[0])
    return index, candidate, rank


def recovery_is_better(
    original: TrackResult | None,
    recovered: TrackResult,
    reference_area: float,
) -> bool:
    if original is None or reference_area <= 0.0:
        return True
    original_area = max(1.0, box_area(original.obj.box_xyxy))
    recovered_area = max(1.0, box_area(recovered.obj.box_xyxy))
    original_error = abs(math.log(original_area / reference_area))
    recovered_error = abs(math.log(recovered_area / reference_area))
    return recovered_error <= original_error + 0.05


def frame_is_occluded(frame: FrameDetections) -> bool:
    """DINO 박스만으로 '제품이 손을 가림' 을 판정한다. 정답 없이 계산된다.

    제품 박스 중심이 손 박스 안에 들어가면 손가락-제품-손바닥 구성으로 본다.
    검수 정답과 대조한 일치율이 79~91% 로, 면적비 기준(31~71%)보다 훨씬 안정적이다.
    손·제품 크기에 무관한 기하 조건이라 상품이 바뀌어도 유지된다.
    """
    hand = frame.best_hand
    obj = frame.best_object
    if hand is None or obj is None:
        return False
    cx = (obj.box_xyxy[0] + obj.box_xyxy[2]) / 2.0
    cy = (obj.box_xyxy[1] + obj.box_xyxy[3]) / 2.0
    x1, y1, x2, y2 = hand.box_xyxy
    return x1 <= cx <= x2 and y1 <= cy <= y2


def track_role_dual(
    model,
    state,
    files: list[Path],
    frames: list[FrameDetections],
    role: str,
    role_text: str,
    args: argparse.Namespace,
) -> dict[int, TrackResult]:
    """두 축으로 시드를 나눈다.

    시간축(전반/후반) — 촬영 규약상 전반은 오른손, 후반은 왼손이다. 손이 바뀌는
        지점에서 끊어야 하나의 시드가 다른 손까지 끌고 가지 않는다.
        이 축은 연속 구간이라 SAM3 의 범위 전파를 그대로 쓴다.

    겹침축(제품감쌈/비감쌈) — 두 상태가 서로 다른 '손의 범위' 를 요구한다는 것이
        6측정에서 반복 확인됐다. 다만 이 프레임들은 시간상 흩어져 있어 범위 전파로
        묶을 수 없다. 그래서 각 시간 구간마다 두 시드를 각각 그 구간 전체에 전파하고,
        프레임마다 그 프레임의 상태에 맞는 쪽을 고른다.

    결과적으로 시간 구간 수 x 2 회 전파한다(구간이 2개면 총 4회, 각각 절반 길이).
    """
    halves = max(1, getattr(args, "hand_seed_halves", 2))
    total = len(files)
    size = -(-total // halves)
    # 시드를 고르는 범위를 구간의 바깥쪽으로 좁힌다.
    # 촬영 규약상 전반은 오른손, 후반은 왼손인데, 지금은 구간 안에서 점수가 가장 높은
    # 프레임을 고르다 보니 손이 크게 나오는 중반(경계 부근)에 몰린다. 실측으로 시드의
    # 32% 가 영상의 40~60% 구간에서 뽑혔고, 100001 은 전반 48% / 후반 53% 로 5%p 밖에
    # 안 떨어져 두 시드가 같은 손이었다. margin 만큼 안쪽을 버리면 두 손이 분리된다.
    margin = max(0.0, min(0.45, getattr(args, "hand_seed_margin", 0.0)))
    results: dict[int, TrackResult] = {}
    for h in range(halves):
        start = h * size
        end = min(start + size - 1, total - 1)
        if start > end:
            break
        pick_start, pick_end = start, end
        # 영상 양 끝은 손이 들어오거나 나가는 중이라 시드로 나쁘다. 실측(시드 64개):
        # 맨 앞 10% 통과율 56.3%, 맨 뒤 10% 38.7% 인데 70~90% 구간은 86.1% 였다.
        # 안쪽(경계) 배제와 달리 이쪽은 바깥을 자른다.
        ends = max(0.0, min(0.45, getattr(args, "hand_seed_endcut", 0.0)))
        if ends > 0:
            cut = int(total * ends)
            pick_start = max(pick_start, cut)
            pick_end = min(pick_end, total - 1 - cut)
            if pick_start > pick_end:                 # 구간이 통째로 잘리면 원복
                pick_start, pick_end = start, end
        if margin > 0 and halves > 1:
            cut = int(size * margin)
            if h == 0:
                pick_end = max(start, end - cut)          # 전반은 앞쪽에서
            elif h == halves - 1:
                pick_start = min(end, start + cut)        # 후반은 뒤쪽에서
            else:
                pick_start, pick_end = start + cut // 2, end - cut // 2
        passes: dict[str, dict[int, TrackResult]] = {}
        for label, keep in (("occ", frame_is_occluded),
                            ("clr", lambda f: not frame_is_occluded(f))):
            cap = None
            factor = getattr(args, "hand_seed_area_factor", 0.0)
            if factor > 0:
                areas = [box_area(f.best_hand.box_xyxy)
                         for i, f in enumerate(frames)
                         if pick_start <= i <= pick_end and f.best_hand is not None and keep(f)]
                if areas:
                    cap = factor * statistics.median(areas)
            fmin = float(getattr(args, "hand_seed_fill_min", 0.0))
            fmap = load_fill(getattr(args, "hand_seed_fill_file", "")) if fmin > 0 else None
            anchor = select_anchor(files, frames, role, args, pick_start, pick_end, keep,
                                   area_cap=cap, whole_bonus=(label == "clr"),
                                   fill=fmap, fill_min=fmin)
            if anchor is None and fmin > 0:          # 채움비 조건이 너무 빡세면 푼다
                anchor = select_anchor(files, frames, role, args, pick_start, pick_end, keep,
                                       area_cap=cap, whole_bonus=(label == "clr"))
            if anchor is None:                       # 좁힌 범위에 없으면 구간 전체에서
                anchor = select_anchor(files, frames, role, args, start, end, keep,
                                       whole_bonus=(label == "clr"))
            if anchor is None:
                print(f"[video] {role} half={h} {label} 시드 없음")
                continue
            passes[label] = track_role_pass(
                model, state, files, frames, anchor, role, role_text, args,
                start_index=start, end_index=end, source=f"sam3_dual_{label}",
            )
        if not passes:
            continue
        picked = {"occ": 0, "clr": 0, "fallback": 0}
        for index in range(start, end + 1):
            want = "occ" if frame_is_occluded(frames[index]) else "clr"
            value = passes.get(want, {}).get(index)
            if value is not None:
                picked[want] += 1
            else:                       # 그 상태의 시드가 없거나 그 프레임을 못 낸 경우
                for other in passes.values():
                    value = other.get(index)
                    if value is not None:
                        picked["fallback"] += 1
                        break
            if value is not None:
                results[index] = value
        print(f"[video] {role} half={h} range={start}:{end} pick={pick_start}:{pick_end} "
              f"seeds={sorted(passes)} "
              f"picked occ={picked['occ']} clr={picked['clr']} fallback={picked['fallback']}")
    return results


def segment_plans(
    files: list[Path],
    frames: list[FrameDetections],
    role: str,
    every: int,
    args: argparse.Namespace,
) -> list[tuple[tuple[int, Candidate, float], int, int]]:
    """구간마다 그 구간 안에서 가장 좋은 프레임을 시드로 잡는다.

    시드 후보가 없는 구간(임계값 미달)은 버리지 않고 직전 구간의 범위에 붙인다.
    그래야 라벨이 비는 프레임이 생기지 않는다. 앞쪽에 아직 시드가 없으면
    뒤에 처음 잡히는 구간이 그 앞까지 맡는다.
    """
    total = len(files)
    plans: list[list] = []
    pending_start: int | None = None
    for start in range(0, total, every):
        end = min(start + every - 1, total - 1)
        anchor = select_anchor(files, frames, role, args, start, end)
        if anchor is None:
            if plans:
                plans[-1][2] = end          # 직전 구간이 이어서 맡는다
            elif pending_start is None:
                pending_start = start
            continue
        if pending_start is not None:
            start = pending_start
            pending_start = None
        plans.append([anchor, start, end])
    return [(a, s, e) for a, s, e in plans]


def track_role_segmented(
    model,
    state,
    files: list[Path],
    frames: list[FrameDetections],
    role: str,
    role_text: str,
    every: int,
    args: argparse.Namespace,
) -> dict[int, TrackResult]:
    plans = segment_plans(files, frames, role, every, args)
    print(f"[video] {role} segmented seeding every={every} segments={len(plans)}")
    if not plans:
        print(f"[video] no reliable {role} seed in any segment")
        return {}
    results: dict[int, TrackResult] = {}
    for anchor, start, end in plans:
        part = track_role_pass(
            model, state, files, frames, anchor, role, role_text, args,
            start_index=start, end_index=end, source="sam3_video_segment",
        )
        # 구간 밖으로 새어 나온 결과는 버린다. 각 프레임은 자기 구간의 시드만 따른다.
        for index, value in part.items():
            if start <= index <= end:
                results[index] = value
    return results


def track_role(
    model,
    video_dir: Path,
    files: list[Path],
    frames: list[FrameDetections],
    anchor: tuple[int, Candidate, float] | None,
    role: str,
    args: argparse.Namespace,
) -> dict[int, TrackResult]:
    if anchor is None:
        print(f"[video] no reliable {role} seed; DINO fallback only")
        return {}

    role_text = (
        args.sam3_hand_prompt
        if role == "hand"
        else next(
            (
                prompt
                for prompt in args.object_prompts
                if prompt.lower() in {"bottle", "can", "box", "bag", "cup", "product package"}
            ),
            "packaged retail product",
        )
    )
    state = model.init_state(
        resource_path=str(video_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=args.video_offload_state_to_cpu,
        async_loading_frames=True,
    )
    try:
        if role == "hand" and getattr(args, "hand_seed_dual", False):
            return track_role_dual(model, state, files, frames, role, role_text, args)
        every = getattr(args, "hand_seed_every", 0) if role == "hand" else 0
        if every > 0:
            return track_role_segmented(
                model, state, files, frames, role, role_text, every, args
            )
        results = track_role_pass(
            model,
            state,
            files,
            frames,
            anchor,
            role,
            role_text,
            args,
        )
        if role == "hand" and args.hand_recovery_max_anchors > 0:
            collapsed_indices = hand_collapse_indices(results, frames, args)
            collapsed = set(collapsed_indices)
            if collapsed:
                print(
                    f"[video] hand collapse frames={len(collapsed)} "
                    f"spans={len(grouped_indices(collapsed_indices, args.hand_recovery_merge_gap))}"
                )
            used = {anchor[0]}
            recovered_indices: set[int] = set()
            for span in grouped_indices(collapsed_indices, args.hand_recovery_merge_gap):
                if len(used) - 1 >= args.hand_recovery_max_anchors:
                    break
                with Image.open(files[anchor[0]]) as image:
                    width, height = image.size
                recovery_anchor = select_hand_recovery_anchor(
                    frames,
                    results,
                    span,
                    collapsed,
                    used,
                    width,
                    height,
                    args,
                )
                if recovery_anchor is None:
                    continue
                used.add(recovery_anchor[0])
                recovery_start = max(0, span[0] - args.hand_recovery_context_frames)
                recovery_end = min(
                    len(files) - 1,
                    span[1] + args.hand_recovery_context_frames,
                )
                recovery_start = min(recovery_start, recovery_anchor[0])
                recovery_end = max(recovery_end, recovery_anchor[0])
                recovered = track_role_pass(
                    model,
                    state,
                    files,
                    frames,
                    recovery_anchor,
                    role,
                    role_text,
                    args,
                    start_index=recovery_start,
                    end_index=recovery_end,
                    source="sam3_video_reseed",
                )
                for index in range(span[0], span[1] + 1):
                    candidate = recovered.get(index)
                    if candidate is None:
                        continue
                    reference_area = local_hand_reference_area(
                        results,
                        index,
                        collapsed,
                        args.hand_recovery_search_radius,
                    )
                    if recovery_is_better(results.get(index), candidate, reference_area):
                        results[index] = candidate
                        recovered_indices.add(index)

            for index in collapsed - recovered_indices:
                candidate = frames[index].best_hand
                if not hand_candidate_is_whole(candidate):
                    continue
                fallback = candidate_fallback(candidate)
                reference_area = local_hand_reference_area(
                    results,
                    index,
                    collapsed,
                    args.hand_recovery_search_radius,
                )
                if fallback is not None and recovery_is_better(
                    results.get(index), fallback, reference_area
                ):
                    fallback.source = "dino_hand_collapse_recovery"
                    results[index] = fallback
            if collapsed:
                print(
                    f"[video] hand collapse recovered={len(recovered_indices)}/"
                    f"{len(collapsed)} reseed_anchors={max(0, len(used) - 1)}"
                )
    finally:
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[video] {role} tracked frames={len(results)}/{len(files)}")
    return results

def candidate_fallback(candidate: Candidate | None) -> TrackResult | None:
    if candidate is None:
        return None
    return TrackResult(
        FinalObject(
            candidate.class_id,
            candidate.class_name,
            candidate.box_xyxy,
            candidate.score,
            candidate.prompt,
        ),
        None,
        "dino_fallback",
        candidate.score,
    )


def eligible_hand_fallback(
    candidate: Candidate | None,
    args: argparse.Namespace,
) -> Candidate | None:
    if getattr(args, "finger_seed_only", False) and hand_candidate_is_detail(candidate):
        return None
    return candidate


def masks_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None or first.shape != second.shape:
        return 0.0
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection) / max(1.0, float(union))


def resolve_cross_class_duplicate(
    hand: TrackResult | None,
    product: TrackResult | None,
    frame: FrameDetections,
    args: argparse.Namespace,
) -> tuple[TrackResult | None, TrackResult | None]:
    if hand is None or product is None:
        return hand, product
    hand_candidate = Candidate(
        HAND_CLASS_ID, "hand", hand.obj.box_xyxy, hand.output_score, hand.obj.prompt
    )
    product_candidate = Candidate(
        args.product_class_id,
        args.product_name,
        product.obj.box_xyxy,
        product.output_score,
        product.obj.prompt,
    )
    box_duplicate = cross_class_duplicate(
        hand_candidate,
        product_candidate,
        args.cross_class_duplicate_iou,
        args.cross_class_duplicate_area_similarity,
    )
    mask_duplicate = masks_iou(hand.mask, product.mask) >= args.cross_class_duplicate_mask_iou
    if not box_duplicate or (hand.mask is not None and product.mask is not None and not mask_duplicate):
        return hand, product

    hand_evidence = frame.best_hand.score if frame.best_hand is not None else hand.output_score
    product_evidence = (
        frame.best_object.score if frame.best_object is not None else product.output_score
    )
    if hand_evidence > product_evidence + args.cross_class_hand_margin:
        return hand, None
    return None, product



def track_foreground_fraction(
    track: TrackResult,
    foreground_mask: np.ndarray | None,
) -> float:
    if foreground_mask is None:
        return 1.0
    if track.mask is not None and track.mask.shape == foreground_mask.shape:
        pixels = np.asarray(track.mask, dtype=bool)
        count = int(pixels.sum())
        if count:
            return float(np.logical_and(pixels, foreground_mask).sum()) / count
    return box_foreground_fraction(foreground_mask, track.obj.box_xyxy)


def foreground_supported_track(
    track: TrackResult | None,
    foreground_mask: np.ndarray | None,
    min_fraction: float,
) -> TrackResult | None:
    if track is None or foreground_mask is None:
        return track
    return track if track_foreground_fraction(track, foreground_mask) >= min_fraction else None

def write_outputs(
    files: list[Path],
    frames: list[FrameDetections],
    hand_tracks: dict[int, TrackResult],
    product_tracks: dict[int, TrackResult],
    args: argparse.Namespace,
) -> tuple[int, int]:
    detect_non_empty = 0
    seg_non_empty = 0
    for index, image_path in enumerate(files):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        frame = frames[index]
        # Detail prompts may seed SAM3, but never become the final DINO-only
        # box when role separation is enabled.
        fallback_hand = eligible_hand_fallback(frame.fallback_hand, args)
        hand = hand_tracks.get(index) or candidate_fallback(fallback_hand)
        product = None
        if not getattr(args, "hand_only", False):
            product = product_tracks.get(index) or candidate_fallback(
                frame.fallback_object
            )
        hand = foreground_supported_track(
            hand, frame.foreground_mask, args.background_min_hand_foreground
        )
        product = foreground_supported_track(
            product, frame.foreground_mask, args.background_min_object_foreground
        )
        hand, product = resolve_cross_class_duplicate(hand, product, frame, args)
        shoe_support = temporal_competitor_support(
            frames, index, "shoe", width, height, args
        )
        arm_support = temporal_competitor_support(
            frames, index, "arm", width, height, args
        )
        shoe_guard = competitor_guard_metrics(
            hand, frame, frame.best_shoe, shoe_support, args
        )
        arm_guard = competitor_guard_metrics(
            hand, frame, frame.best_arm, arm_support, args
        )
        rejected_by = None
        if hand is not None:
            if args.shoe_competitor_mode == "reject" and shoe_guard["would_reject"]:
                hand = None
                rejected_by = "shoe"
            elif args.arm_competitor_mode == "reject" and arm_guard["would_reject"]:
                hand = None
                rejected_by = "arm"
        selected = [item for item in (hand, product) if item is not None]
        final_objects = [item.obj for item in selected]

        detect_path = args.out_label_dir / f"{image_path.stem}.txt"
        segment_path = args.out_seg_label_dir / f"{image_path.stem}.txt"
        visualized_path = args.out_visualized_dir / f"{image_path.stem}{image_path.suffix.lower()}"
        write_detect_label(detect_path, final_objects, width, height)
        seg_count = write_segment_label(
            segment_path,
            final_objects,
            width,
            height,
            args.segmentation_fallback,
        )
        if args.save_visualized or args.retain_all_frames:
            draw_visualized(image, final_objects, visualized_path)
        if final_objects:
            detect_non_empty += 1
        if seg_count:
            seg_non_empty += 1

        write_jsonl(
            args.out_debug_jsonl,
            {
                "image": image_path.name,
                "backend": "grounding_dino_plus_sam3_video",
                "visible_hand_mode": args.visible_hand_mode,
                "finger_seed_only": args.finger_seed_only,
                "background_reference": frame.background_reference,
                "hand_candidates": [asdict(item) for item in frame.hand_candidates],
                "object_candidates": [asdict(item) for item in frame.object_candidates],
                "shoe_candidates": [asdict(item) for item in frame.shoe_candidates],
                "arm_candidates": [asdict(item) for item in frame.arm_candidates],
                "best_hand": asdict(frame.best_hand) if frame.best_hand else None,
                "best_object": asdict(frame.best_object) if frame.best_object else None,
                "competitor_guard": {
                    "shoe_mode": args.shoe_competitor_mode,
                    "arm_mode": args.arm_competitor_mode,
                    "shoe": shoe_guard,
                    "arm": arm_guard,
                    "rejected_by": rejected_by,
                },
                "hand_source": hand.source if hand else None,
                "product_source": product.source if product else None,
                "final": [asdict(item) for item in final_objects],
            },
        )
    return detect_non_empty, seg_non_empty


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_label_dir", required=True)
    parser.add_argument("--out_seg_label_dir", required=True)
    parser.add_argument("--out_visualized_dir", required=True)
    parser.add_argument("--out_debug_jsonl", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument(
        "--hand_only",
        action="store_true",
        help="Detect and track class 0 hand only; skip all product inference and output.",
    )
    parser.add_argument("--product_class_id", type=int)
    parser.add_argument("--product_name", default="")
    parser.add_argument("--product_prompts", default="")
    parser.add_argument(
        "--hand_prompts",
        default=(
            "complete human hand|hand from wrist to fingertips|grasping human hand|"
            "left human hand|right human hand|palm|fingers"
        ),
    )
    parser.add_argument(
        "--sam3_hand_prompt",
        default="complete human hand from wrist to fingertips",
    )
    parser.add_argument("--dino_model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--dino_device", default="cuda")
    parser.add_argument("--sam3_device", default="cuda")
    parser.add_argument("--hand_box_threshold", type=float, default=0.25)
    parser.add_argument("--object_box_threshold", type=float, default=0.22)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--sam3_confidence", type=float, default=0.35)
    parser.add_argument("--sam3_autocast_dtype", choices=["bfloat16", "float16", "none"], default="bfloat16")
    parser.add_argument("--sam3_area_weight", type=float, default=0.15)
    parser.add_argument("--sam3_min_prompt_iou", type=float, default=0.15)
    parser.add_argument("--sam3_min_area_ratio", type=float, default=0.15)
    parser.add_argument("--sam3_max_area_ratio", type=float, default=4.0)
    parser.add_argument("--temporal_object_iou_weight", type=float, default=0.20)
    parser.add_argument("--temporal_decay", type=float, default=0.75)
    parser.add_argument("--temporal_max_gap", type=int, default=5)
    parser.add_argument("--min_temporal_compatibility", type=float, default=0.30)
    parser.add_argument("--min_hand_object_relation", type=float, default=0.08)
    parser.add_argument("--object_solo_threshold", type=float, default=0.38)
    parser.add_argument("--hand_solo_threshold", type=float, default=0.40)
    parser.add_argument("--segmentation_fallback", choices=["skip", "box"], default="skip")
    parser.add_argument("--max_hand_candidates", type=int, default=8)
    parser.add_argument("--max_object_candidates", type=int, default=10)
    parser.add_argument("--nms_iou", type=float, default=0.55)
    parser.add_argument("--min_hand_area", type=float, default=0.0)
    parser.add_argument("--max_hand_area", type=float, default=1.0)
    parser.add_argument("--min_object_area", type=float, default=0.0)
    parser.add_argument("--max_object_area", type=float, default=1.0)
    parser.add_argument("--min_hand_aspect", type=float, default=0.0)
    parser.add_argument("--max_hand_aspect", type=float, default=1e9)
    parser.add_argument("--min_object_aspect", type=float, default=0.0)
    parser.add_argument("--max_object_aspect", type=float, default=1e9)
    parser.add_argument("--max_hand_width", type=float, default=1.0)
    parser.add_argument("--max_hand_height", type=float, default=1.0)
    parser.add_argument("--max_object_width", type=float, default=1.0)
    parser.add_argument("--max_object_height", type=float, default=1.0)
    parser.add_argument("--polygon_epsilon", type=float, default=0.003)
    parser.add_argument("--min_polygon_points", type=int, default=3)
    parser.add_argument("--video_product_seed_threshold", type=float, default=0.24)
    parser.add_argument("--video_hand_seed_threshold", type=float, default=0.30)
    parser.add_argument("--video_min_output_score", type=float, default=0.40)
    parser.add_argument("--video_seed_support_radius", type=int, default=3)
    parser.add_argument("--video_offload_state_to_cpu", action="store_true", default=True)
    parser.add_argument("--no_video_offload_state_to_cpu", dest="video_offload_state_to_cpu", action="store_false")
    parser.add_argument("--cross_class_duplicate_iou", type=float, default=0.88)
    parser.add_argument("--cross_class_duplicate_area_similarity", type=float, default=0.85)
    parser.add_argument("--cross_class_duplicate_mask_iou", type=float, default=0.80)
    parser.add_argument("--cross_class_hand_margin", type=float, default=0.10)
    parser.add_argument("--background_reference", action="append", default=[])
    parser.add_argument("--background_pixel_threshold", type=float, default=22.0)
    parser.add_argument("--background_min_hand_foreground", type=float, default=0.06)
    parser.add_argument("--background_min_object_foreground", type=float, default=0.08)
    parser.add_argument("--compact_hand_max_product_gap", type=float, default=0.18)
    parser.add_argument("--compact_hand_cluster_radius", type=float, default=0.22)
    parser.add_argument("--compact_hand_score_window", type=float, default=0.14)
    parser.add_argument("--compact_hand_min_margin", type=float, default=0.08)
    parser.add_argument("--compact_hand_expansion", type=float, default=0.30)
    parser.add_argument("--hand_mask_support_expansion", type=float, default=0.15)
    parser.add_argument("--hand_partial_mask_support_expansion", type=float, default=0.45)
    parser.add_argument("--hand_max_mask_support_expansion", type=float, default=0.60)
    parser.add_argument("--hand_component_min_area_ratio", type=float, default=0.04)
    parser.add_argument("--hand_component_max_gap", type=float, default=0.08)
    parser.add_argument("--hand_component_product_gap", type=float, default=0.05)
    parser.add_argument("--hand_collapse_area_ratio", type=float, default=0.45)
    parser.add_argument("--hand_collapse_dimension_ratio", type=float, default=0.50)
    parser.add_argument("--hand_collapse_history_min", type=int, default=4)
    parser.add_argument("--hand_collapse_history_window", type=int, default=9)
    parser.add_argument("--hand_recovery_min_area_ratio", type=float, default=0.65)
    parser.add_argument("--hand_recovery_search_radius", type=int, default=60)
    parser.add_argument("--hand_recovery_context_frames", type=int, default=45)
    parser.add_argument("--hand_recovery_max_anchors", type=int, default=0)
    # 0 이면 영상 전체에 시드 1개(기존 동작). N>0 이면 N 프레임마다 시드를 다시 잡는다.
    parser.add_argument("--hand_seed_every", type=int, default=0)
    # 전반/후반 + 겹침/비겹침 두 축으로 시드를 나눈다
    parser.add_argument("--hand_seed_dual", action="store_true")
    parser.add_argument("--hand_seed_halves", type=int, default=2)
    # 구간의 안쪽(경계 쪽) 이 비율만큼을 시드 탐색에서 제외한다. 0.20 이면 40/40 이다.
    parser.add_argument("--hand_seed_margin", type=float, default=0.0)
    # 영상 앞뒤 이 비율을 시드 탐색에서 제외 (0.10 이면 앞뒤 10% 씩)
    parser.add_argument("--hand_seed_endcut", type=float, default=0.0)
    # 2패스: 1차 run 의 채움비 파일로 시드 후보를 거른다
    parser.add_argument("--hand_seed_fill_file", default="")
    parser.add_argument("--hand_seed_fill_min", type=float, default=0.0)
    # 1차 run 의 detections.jsonl 로 DINO 스캔을 대체 (2패스 전용)
    parser.add_argument("--dino_cache", default="")
    parser.add_argument("--product_cache", default="")
    parser.add_argument("--dino_batch", type=int, default=1)
    parser.add_argument("--dino_autocast_dtype",
                        choices=["bfloat16", "float16", "none"], default="none")
    # 0 이면 끔. F>0 이면 구간 중앙값의 F 배를 넘는 후보를 시드에서 뺀다.
    parser.add_argument("--hand_seed_area_factor", type=float, default=0.0)
    # 시드에 점 프롬프트를 덧대 교정한다 (기본 OFF)
    parser.add_argument("--hand_point_refine", action="store_true")
    parser.add_argument("--hand_point_arm_negative", action="store_true", default=True)
    parser.add_argument("--hand_point_arm_offset", type=float, default=0.35)
    parser.add_argument("--hand_recovery_merge_gap", type=int, default=5)
    parser.add_argument(
        "--visible_hand_mode",
        action="store_true",
        help=(
            "Experimental policy: annotate only currently visible hand pixels; "
            "finger-only observations are valid and are not expanded to an occluded full hand."
        ),
    )
    parser.add_argument("--visible_hand_support_expansion", type=float, default=0.08)
    parser.add_argument("--visible_hand_min_dino_overlap", type=float, default=0.70)
    parser.add_argument(
        "--finger_seed_only",
        action="store_true",
        help="P1A opt-in: finger/palm detail proposals may seed SAM3 but cannot be final DINO fallback boxes.",
    )
    parser.add_argument(
        "--shoe_competitor_mode",
        choices=("off", "shadow", "reject"),
        default="off",
        help="P1B/P1C opt-in. Shadow only logs decisions; reject applies the guarded veto.",
    )
    parser.add_argument(
        "--arm_competitor_mode",
        choices=("off", "shadow", "reject"),
        default="off",
        help="P1D opt-in. Keep at shadow until arm behavior is visually validated.",
    )
    parser.add_argument(
        "--shoe_competitor_prompts",
        default="shoe|sneaker|footwear|lower leg and shoe",
    )
    parser.add_argument(
        "--arm_competitor_prompts",
        default="human forearm|human arm|sleeve|lower arm",
    )
    parser.add_argument("--competitor_box_threshold", type=float, default=0.25)
    parser.add_argument("--competitor_text_threshold", type=float, default=0.20)
    parser.add_argument("--max_competitor_candidates", type=int, default=10)
    parser.add_argument("--competitor_min_score", type=float, default=0.40)
    parser.add_argument("--competitor_score_margin", type=float, default=0.05)
    parser.add_argument("--competitor_min_overlap", type=float, default=0.50)
    parser.add_argument("--competitor_temporal_radius", type=int, default=2)
    parser.add_argument("--competitor_min_temporal_support", type=int, default=1)
    parser.add_argument("--competitor_temporal_compatibility", type=float, default=0.25)
    parser.add_argument(
        "--dino_only",
        action="store_true",
        help="Skip SAM3 video tracking for fast prompt/candidate pilots.",
    )
    parser.add_argument("--save_visualized", action="store_true", default=True)
    parser.add_argument("--no_save_visualized", dest="save_visualized", action="store_false")
    parser.add_argument("--retain_all_frames", action="store_true", default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    read_hf_tokens_from_env_file(project_root / ".env")

    img_dir = Path(args.img_dir)
    product_folder = img_dir.parent.name
    try:
        default_class_id = int(product_folder.split("_", 1)[0])
    except Exception:
        default_class_id = 1
    args.product_class_id = int(args.product_class_id or default_class_id)
    args.product_name = args.product_name or strip_folder_timestamp(product_folder)
    args.hand_prompts = split_prompts(args.hand_prompts)
    args.object_prompts = []
    if not getattr(args, "hand_only", False):
        args.object_prompts = (
            split_prompts(args.product_prompts) or product_prompts_from_folder(product_folder)
        )
    args.shoe_competitor_prompts = split_prompts(args.shoe_competitor_prompts)
    args.arm_competitor_prompts = split_prompts(args.arm_competitor_prompts)
    args.out_label_dir = Path(args.out_label_dir)
    args.out_seg_label_dir = Path(args.out_seg_label_dir)
    args.out_visualized_dir = Path(args.out_visualized_dir)
    args.out_debug_jsonl = Path(args.out_debug_jsonl)
    args.summary_path = Path(args.summary_path)
    args.background_references = load_background_references(args.background_reference)

    for directory in (
        args.out_label_dir,
        args.out_seg_label_dir,
        args.out_visualized_dir,
        args.out_debug_jsonl.parent,
        args.summary_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if args.out_debug_jsonl.exists():
        args.out_debug_jsonl.unlink()

    files = image_files(img_dir)
    if args.start_index > 0:
        files = files[args.start_index :]
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"[error] no images found: {img_dir}", file=sys.stderr)
        return 2

    print(f"[info] backend=grounding_dino_plus_sam3_video images={len(files)}")
    print(f"[info] hand_prompts={args.hand_prompts}")
    print(f"[info] object_prompts={args.object_prompts}")
    print(
        f"[info] experimental finger_seed_only={args.finger_seed_only} "
        f"shoe_competitor={args.shoe_competitor_mode} "
        f"arm_competitor={args.arm_competitor_mode}"
    )
    print(f"[info] background_references={[name for name, _ in args.background_references]}")
    started = time.time()

    set_dino_autocast(getattr(args, 'dino_autocast_dtype', 'none'))
    dino_processor, dino_model = load_grounding_dino(args.dino_model, args.dino_device)
    cache = getattr(args, "dino_cache", "")
    stage = {"dino": 0.0, "sam3_load": 0.0, "product": 0.0, "hand": 0.0, "write": 0.0}
    mark = time.time()
    if cache and Path(cache).is_file():
        frames = load_dino_cache(cache, files, args)
    else:
        frames = scan_with_dino(files, dino_processor, dino_model, args)
    stage["dino"] = time.time() - mark
    object_anchor = (
        None if getattr(args, "hand_only", False) else select_anchor(files, frames, "object", args)
    )
    hand_anchor = select_anchor(files, frames, "hand", args)
    del dino_model, dino_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hand_tracks: dict[int, TrackResult] = {}
    product_tracks: dict[int, TrackResult] = {}
    product_cache = getattr(args, "product_cache", "")
    product_cached = bool(product_cache) and Path(product_cache).is_file()
    if product_cached:
        product_tracks = load_product_cache(product_cache)
    video_error = ""
    if args.dino_only:
        print("[info] dino_only=True; skipping SAM3 video tracking")
    elif object_anchor is not None or hand_anchor is not None:
        try:
            print("[info] loading SAM3 video model")
            mark = time.time()
            model = build_sam3_video_model(
                load_from_HF=True,
                apply_temporal_disambiguation=True,
                device=args.sam3_device,
                compile=False,
            )
            model.eval()
            stage["sam3_load"] = time.time() - mark
            with ordered_video_directory(files) as video_dir:
                if not getattr(args, "hand_only", False) and not product_cached:
                    mark = time.time()
                    product_tracks = track_role(
                        model, video_dir, files, frames, object_anchor, "object", args
                    )
                    stage["product"] = time.time() - mark
                    if product_cache:
                        save_product_cache(Path(product_cache), product_tracks)
                mark = time.time()
                hand_tracks = track_role(
                    model, video_dir, files, frames, hand_anchor, "hand", args
                )
                stage["hand"] = time.time() - mark
            del model
        except Exception as exc:
            video_error = f"{type(exc).__name__}: {exc}"
            print(f"[warn] SAM3 video tracking failed; using DINO fallback: {video_error}", file=sys.stderr)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mark = time.time()
    detect_non_empty, seg_non_empty = write_outputs(
        files,
        frames,
        hand_tracks,
        product_tracks,
        args,
    )
    stage["write"] = time.time() - mark
    summary = {
        "backend": "grounding_dino_plus_sam3_video",
        "hand_only": getattr(args, "hand_only", False),
        "visible_hand_mode": args.visible_hand_mode,
        "finger_seed_only": args.finger_seed_only,
        "shoe_competitor_mode": args.shoe_competitor_mode,
        "arm_competitor_mode": args.arm_competitor_mode,
        "dino_only": args.dino_only,
        "images": len(files),
        "detect_non_empty": detect_non_empty,
        "detect_empty": len(files) - detect_non_empty,
        "seg_non_empty": seg_non_empty,
        "seg_empty": len(files) - seg_non_empty,
        "product_seed_frame": object_anchor[0] if object_anchor else -1,
        "product_seed_score": object_anchor[1].score if object_anchor else 0.0,
        "hand_seed_frame": hand_anchor[0] if hand_anchor else -1,
        "hand_seed_score": hand_anchor[1].score if hand_anchor else 0.0,
        "product_tracked_frames": len(product_tracks),
        "hand_tracked_frames": len(hand_tracks),
        "video_error": video_error,
        "background_references": [name for name, _ in args.background_references],
        "dino_sec": round(stage["dino"], 1),
        "sam3_load_sec": round(stage["sam3_load"], 1),
        "product_sec": round(stage["product"], 1),
        "hand_sec": round(stage["hand"], 1),
        "write_sec": round(stage["write"], 1),
        "dino_cached": bool(cache and Path(cache).is_file()),
        "product_cached": product_cached,
        "elapsed_sec": round(time.time() - started, 3),
    }
    args.summary_path.write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
