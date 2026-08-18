#!/usr/bin/env python3
"""Grounding DINO + SAM3 automatic annotation for YOLO training.

The pipeline is image based:
1. Grounding DINO proposes hand/product boxes.
2. SAM3 refines each selected box into a mask using a geometric prompt.
3. YOLO detection labels are written to ``labels``.
4. YOLO segmentation labels are written to ``labels_seg``.

No local SAM2 package or checkpoint is imported by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
HAND_CLASS_ID = 0


@dataclass
class Candidate:
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    score: float
    prompt: str
    source: str = "dino"


@dataclass
class FinalObject:
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    score: float
    prompt: str
    mask_iou: float = 0.0
    mask_score: float = 0.0
    mask_area_ratio: float = 0.0
    mask_component_count: int = 0
    mask_valid: bool = False
    polygon_points: list[tuple[float, float]] | None = None


def read_hf_tokens_from_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for line in env_file.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            continue
        os.environ.setdefault(key, value.strip().strip("'").strip('"'))


def image_files(img_dir: Path) -> list[Path]:
    direct = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if direct:
        return direct
    nested = img_dir / "images"
    if nested.is_dir():
        return sorted(p for p in nested.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []


def class_id_from_folder(folder_name: str) -> int:
    try:
        return int(folder_name.split("_", 1)[0])
    except Exception:
        return 1


def strip_folder_timestamp(folder_name: str) -> str:
    name = re.sub(r"_\d{8}_\d{6}$", "", folder_name)
    return re.sub(r"^\d+_", "", name)


def product_prompts_from_folder(folder_name: str) -> list[str]:
    upper = folder_name.upper()
    if "BOTTLE" in upper:
        shape = "bottle"
    elif "CAN" in upper:
        shape = "can"
    elif "BOX" in upper:
        shape = "box"
    elif "BAG" in upper:
        shape = "bag"
    else:
        shape = "product package"

    ignored = {"bag", "box", "bottle", "can", "cup", "v2", "v3", "v4"}
    words: list[str] = []
    for token in folder_name.split("_")[1:]:
        word = token.lower()
        if word in ignored:
            continue
        if re.fullmatch(r"20\d{6}|\d{6}|\d{6,8}", word):
            continue
        if re.fullmatch(r"\d+(g|ml|l)", word):
            continue
        if re.fullmatch(r"v\d+", word):
            continue
        words.append(word)
    phrase = " ".join(words[:5]).strip()

    prompts = []
    if phrase:
        prompts.append(f"{phrase} {shape}")
        prompts.append(phrase)
    prompts.extend([shape, f"packaged {shape}", "packaged retail product"])

    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        key = prompt.lower().strip()
        if key and key not in seen:
            unique.append(prompt)
            seen.add(key)
    return unique


def split_prompts(value: str) -> list[str]:
    prompts = []
    seen = set()
    for raw in re.split(r"[|]", value or ""):
        prompt = raw.strip()
        key = prompt.lower()
        if prompt and key not in seen:
            prompts.append(prompt)
            seen.add(key)
    return prompts


def grounding_text(prompts: Iterable[str]) -> str:
    parts = [p.strip().rstrip(".") for p in prompts if p.strip()]
    return ". ".join(parts) + "."


def clamp_box(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    return inter / max(1e-9, union)


def box_gap_norm(a: tuple[float, float, float, float], b: tuple[float, float, float, float], width: int, height: int) -> float:
    dx = max(0.0, b[0] - a[2], a[0] - b[2]) / max(1.0, width)
    dy = max(0.0, b[1] - a[3], a[1] - b[3]) / max(1.0, height)
    return math.sqrt(dx * dx + dy * dy)


def box_center_distance_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    width: int,
    height: int,
) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot((ax - bx) / max(1.0, width), (ay - by) / max(1.0, height))


def box_area_similarity(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ratio = box_area(a) / max(1e-9, box_area(b))
    return math.exp(-abs(math.log(max(1e-9, ratio))))


def frame_number(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def xyxy_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / width, cy / height, bw / width, bh / height


def xyxy_to_cxcywh_norm(box: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = xyxy_to_yolo(box, width, height)
    return [cx, cy, bw, bh]


def nms(candidates: list[Candidate], iou_threshold: float, key_func=None) -> list[Candidate]:
    kept: list[Candidate] = []
    if key_func is None:
        key_func = lambda item: item.score
    for cand in sorted(candidates, key=key_func, reverse=True):
        if all(box_iou(cand.box_xyxy, prev.box_xyxy) <= iou_threshold for prev in kept):
            kept.append(cand)
    return kept


def shape_ok(
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
    # Keep the legacy parameters for CLI compatibility, but do not reject a
    # detection by absolute size or aspect. Camera distance varies too much for
    # those priors to generalize across products.
    del min_area, max_area, min_aspect, max_aspect
    del max_width_ratio, max_height_ratio
    if width <= 0 or height <= 0 or not all(math.isfinite(value) for value in box):
        return False
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False
    return x2 > 0.0 and y2 > 0.0 and x1 < float(width) and y1 < float(height)


def load_grounding_dino(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


_DINO_AUTOCAST = "none"


def set_dino_autocast(dtype_name: str) -> None:
    """DINO 는 SAM3 와 달리 fp32 로 돌고 있었다. 전체 시간의 30% 를 쓰는 단계라
    여기만 반정밀도로 바꿔도 크게 줄어든다. 박스가 달라질 수 있어 기본값은 끔이다."""
    global _DINO_AUTOCAST
    _DINO_AUTOCAST = dtype_name or "none"


def dino_autocast(device: str):
    if _DINO_AUTOCAST == "none" or not str(device).startswith("cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if _DINO_AUTOCAST == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def detect_with_dino(
    image: Image.Image,
    processor,
    model,
    device: str,
    prompts: list[str],
    class_id: int,
    class_name: str,
    box_threshold: float,
    text_threshold: float,
    max_candidates: int,
) -> list[Candidate]:
    text = grounding_text(prompts)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with dino_autocast(device):
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    return candidates_from_results(
        results, class_id, class_name, text, image.width, image.height, max_candidates
    )


def candidates_from_results(
    results: dict,
    class_id: int,
    class_name: str,
    text: str,
    width: int,
    height: int,
    max_candidates: int,
) -> list[Candidate]:
    """후처리 결과를 Candidate 목록으로 바꾼다.

    배치 추론에서도 그대로 쓰려고 detect_with_dino 에서 떼어냈다. 한 장씩 돌 때와
    같은 함수를 쓰므로 결과가 갈라질 여지가 없다.
    """
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    labels = results.get("labels", [""] * len(boxes))
    candidates: list[Candidate] = []
    for box, score, label in zip(boxes, scores, labels):
        clamped = clamp_box(box.detach().cpu().tolist(), width, height)
        if clamped is None:
            continue
        candidates.append(Candidate(class_id, class_name, clamped, float(score), str(label or text)))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def build_sam3_processor(device: str, confidence_threshold: float) -> Sam3Processor:
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def sam3_autocast(args: argparse.Namespace):
    if not str(args.sam3_device).startswith("cuda") or args.sam3_autocast_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if args.sam3_autocast_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def select_prompt_component(
    mask: np.ndarray,
    prompt_box: tuple[float, float, float, float],
) -> tuple[np.ndarray, int]:
    """Keep the connected mask component best supported by the DINO prompt box."""
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    foreground_count = max(0, component_count - 1)
    if foreground_count <= 1:
        return binary.astype(bool), foreground_count

    x1, y1, x2, y2 = prompt_box
    prompt_area = max(1.0, (x2 - x1) * (y2 - y1))
    best_label = 1
    best_score = -1.0
    for label in range(1, component_count):
        cx, cy, component_width, component_height, area = stats[label]
        component_box = (float(cx), float(cy), float(cx + component_width), float(cy + component_height))
        ix1, iy1 = max(x1, component_box[0]), max(y1, component_box[1])
        ix2, iy2 = min(x2, component_box[2]), min(y2, component_box[3])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        component_coverage = intersection / max(1.0, float(area))
        prompt_coverage = intersection / prompt_area
        score = 0.65 * component_coverage + 0.30 * prompt_coverage + 0.05 * box_iou(prompt_box, component_box)
        if score > best_score:
            best_score = score
            best_label = label
    return labels == best_label, foreground_count


def mask_to_polygon(mask: np.ndarray, width: int, height: int, epsilon_ratio: float, min_points: int) -> list[tuple[float, float]] | None:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4.0:
        return None
    epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < min_points:
        return None
    points = [(float(x) / width, float(y) / height) for x, y in approx]
    return [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in points]


def refine_with_sam3(
    image: Image.Image,
    state: dict,
    sam3_processor: Sam3Processor,
    candidate: Candidate,
    args: argparse.Namespace,
) -> FinalObject:
    width, height = image.size
    sam3_processor.reset_all_prompts(state)
    prompt_box = xyxy_to_cxcywh_norm(candidate.box_xyxy, width, height)
    try:
        with sam3_autocast(args):
            out_state = sam3_processor.add_geometric_prompt(prompt_box, True, state)
        boxes = out_state.get("boxes")
        masks = out_state.get("masks")
        scores = out_state.get("scores")
        if boxes is None or masks is None or scores is None or len(boxes) == 0:
            raise RuntimeError("SAM3 returned no mask")

        masks_np = masks.detach().cpu().numpy()
        scores_np = scores.detach().float().cpu().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        best_mask: np.ndarray | None = None
        best_box: tuple[float, float, float, float] | None = None
        best_idx = -1
        best_rank = -1e9
        best_iou = 0.0
        best_area_ratio = 0.0
        best_component_count = 0
        candidate_area = max(1e-9, box_area(candidate.box_xyxy))
        for idx, raw_mask in enumerate(masks_np):
            cleaned_mask, component_count = select_prompt_component(raw_mask > 0, candidate.box_xyxy)
            refined_box = mask_to_box(cleaned_mask)
            if refined_box is None:
                continue
            refined_box = clamp_box(refined_box, width, height)
            if refined_box is None:
                continue
            iou = box_iou(candidate.box_xyxy, refined_box)
            area_ratio = box_area(refined_box) / candidate_area
            area_score = math.exp(-abs(math.log(max(1e-9, area_ratio))))
            rank = float(scores_np[idx]) + args.sam3_iou_weight * iou + args.sam3_area_weight * area_score
            if rank > best_rank:
                best_rank = rank
                best_idx = idx
                best_mask = cleaned_mask
                best_box = refined_box
                best_iou = iou
                best_area_ratio = area_ratio
                best_component_count = component_count

        if best_idx < 0 or best_mask is None or best_box is None:
            raise RuntimeError("SAM3 returned no valid component")

        mask_valid = (
            best_iou >= args.sam3_min_prompt_iou
            and args.sam3_min_area_ratio <= best_area_ratio <= args.sam3_max_area_ratio
        )
        polygon = (
            mask_to_polygon(best_mask, width, height, args.polygon_epsilon, args.min_polygon_points)
            if mask_valid
            else None
        )
        return FinalObject(
            candidate.class_id,
            candidate.class_name,
            best_box if mask_valid else candidate.box_xyxy,
            candidate.score,
            candidate.prompt,
            mask_iou=best_iou,
            mask_score=float(scores_np[best_idx]),
            mask_area_ratio=best_area_ratio,
            mask_component_count=best_component_count,
            mask_valid=mask_valid,
            polygon_points=polygon,
        )
    except Exception as exc:
        if args.verbose:
            print(f"[warn] SAM3 refine failed for {candidate.class_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return FinalObject(candidate.class_id, candidate.class_name, candidate.box_xyxy, candidate.score, candidate.prompt)


def cross_class_duplicate(
    first: Candidate,
    second: Candidate,
    iou_threshold: float = 0.88,
    area_similarity_threshold: float = 0.85,
) -> bool:
    """Return True when two class predictions describe effectively the same box."""
    return (
        box_iou(first.box_xyxy, second.box_xyxy) >= iou_threshold
        and box_area_similarity(first.box_xyxy, second.box_xyxy) >= area_similarity_threshold
    )


def relation_bonus(candidate: Candidate, others: list[Candidate], width: int, height: int) -> float:
    if not others:
        return 0.0
    best = 0.0
    for other in others:
        iou = box_iou(candidate.box_xyxy, other.box_xyxy)
        gap = box_gap_norm(candidate.box_xyxy, other.box_xyxy, width, height)
        area_similarity = box_area_similarity(candidate.box_xyxy, other.box_xyxy)
        # Contact and partial overlap are useful action cues. Near-identical boxes are
        # usually one region classified as both HAND and PRODUCT, so do not reward them.
        distinct_overlap = iou * (1.0 - area_similarity)
        proximity = 1.0 / (1.0 + 20.0 * gap)
        best = max(best, 0.15 * proximity + 0.12 * distinct_overlap)
    return best


def temporal_compatibility(
    candidate: Candidate,
    previous_object_box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> float:
    if candidate.class_id == HAND_CLASS_ID or previous_object_box is None:
        return 0.0
    iou = box_iou(candidate.box_xyxy, previous_object_box)
    center_score = math.exp(-6.0 * box_center_distance_norm(candidate.box_xyxy, previous_object_box, width, height))
    area_score = box_area_similarity(candidate.box_xyxy, previous_object_box)
    return 0.65 * iou + 0.25 * center_score + 0.10 * area_score


def temporal_object_bonus(
    candidate: Candidate,
    previous_object_box: tuple[float, float, float, float] | None,
    previous_object_age: int,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> float:
    compatibility = temporal_compatibility(candidate, previous_object_box, width, height)
    decay = float(args.temporal_decay) ** max(0, int(previous_object_age))
    return float(args.temporal_object_iou_weight) * compatibility * decay


def object_selection_score(
    candidate: Candidate,
    hands: list[Candidate],
    previous_object_box: tuple[float, float, float, float] | None,
    previous_object_age: int,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> float:
    return (
        candidate.score
        + relation_bonus(candidate, hands, width, height)
        + temporal_object_bonus(
            candidate,
            previous_object_box,
            previous_object_age,
            width,
            height,
            args,
        )
    )


def select_final_candidates(
    candidates: list[Candidate],
    width: int,
    height: int,
    args: argparse.Namespace,
    previous_object_box: tuple[float, float, float, float] | None = None,
    previous_object_age: int = 0,
) -> list[Candidate]:
    hands = [c for c in candidates if c.class_id == HAND_CLASS_ID]
    objects = [c for c in candidates if c.class_id != HAND_CLASS_ID]

    hands = [
        c for c in hands
        if shape_ok(
            c.box_xyxy,
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
    objects = [
        c for c in objects
        if shape_ok(
            c.box_xyxy,
            width,
            height,
            args.min_object_area,
            args.max_object_area,
            args.min_object_aspect,
            args.max_object_aspect,
            args.max_object_width,
            args.max_object_height,
        )
    ]

    hands = nms(hands, args.nms_iou)
    objects = nms(
        objects,
        args.nms_iou,
        key_func=lambda candidate: object_selection_score(
            candidate,
            hands,
            previous_object_box,
            previous_object_age,
            width,
            height,
            args,
        ),
    )

    selected_object: Candidate | None = None
    selected_hand: Candidate | None = None
    if objects:
        selected_object = max(
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
    if hands:
        distinct_hands = [
            hand
            for hand in hands
            if selected_object is None or not cross_class_duplicate(hand, selected_object)
        ]
    else:
        distinct_hands = []
    if distinct_hands:
        selected_hand = max(
            distinct_hands,
            key=lambda candidate: (
                candidate.score
                + relation_bonus(
                    candidate,
                    [selected_object] if selected_object is not None else objects,
                    width,
                    height,
                )
            ),
        )

    selected: list[Candidate] = []
    object_supported = False
    if selected_object is not None:
        relation = relation_bonus(selected_object, [selected_hand] if selected_hand else [], width, height)
        continuity = temporal_compatibility(selected_object, previous_object_box, width, height)
        continuity *= float(args.temporal_decay) ** max(0, int(previous_object_age))
        object_supported = (
            selected_object.score >= args.object_solo_threshold
            or relation >= args.min_hand_object_relation
            or continuity >= args.min_temporal_compatibility
        )
        if object_supported:
            selected.append(selected_object)

    if selected_hand is not None and (
        object_supported or selected_hand.score >= args.hand_solo_threshold
    ):
        selected.append(selected_hand)

    selected.sort(key=lambda candidate: (candidate.class_id != HAND_CLASS_ID, candidate.class_id))
    return selected


def write_detect_label(path: Path, objects: list[FinalObject], width: int, height: int) -> None:
    lines = []
    for obj in objects:
        cx, cy, bw, bh = xyxy_to_yolo(obj.box_xyxy, width, height)
        if bw <= 0.0 or bh <= 0.0:
            continue
        lines.append(f"{obj.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_segment_label(path: Path, objects: list[FinalObject], width: int, height: int, fallback: str) -> int:
    lines = []
    for obj in objects:
        points = obj.polygon_points
        if points is None and fallback == "box":
            x1, y1, x2, y2 = obj.box_xyxy
            points = [(x1 / width, y1 / height), (x2 / width, y1 / height), (x2 / width, y2 / height), (x1 / width, y2 / height)]
        if not points:
            continue
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"{obj.class_id} {coords}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def draw_visualized(image: Image.Image, objects: list[FinalObject], output_path: Path) -> None:
    canvas = image.convert("RGB")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(canvas)
    colors = {HAND_CLASS_ID: (255, 80, 80), "object": (80, 255, 120)}
    for obj in objects:
        color = colors[HAND_CLASS_ID] if obj.class_id == HAND_CLASS_ID else colors["object"]
        x1, y1, x2, y2 = obj.box_xyxy
        if obj.polygon_points:
            pts = [(x * canvas.width, y * canvas.height) for x, y in obj.polygon_points]
            draw_overlay.polygon(pts, fill=(*color, 70))
            draw.line(pts + [pts[0]], fill=color, width=2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 3, max(0, y1 - 14)), f"{obj.class_name}:{obj.score:.2f}", fill=color)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def process_image(
    image_path: Path,
    dino_processor,
    dino_model,
    sam3_processor: Sam3Processor,
    args: argparse.Namespace,
    previous_object_box: tuple[float, float, float, float] | None = None,
    previous_object_age: int = 0,
) -> tuple[int, int, int, int, tuple[float, float, float, float] | None]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    hand_candidates = detect_with_dino(
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
    object_candidates = detect_with_dino(
        image,
        dino_processor,
        dino_model,
        args.dino_device,
        args.object_prompts,
        args.product_class_id,
        args.product_name,
        args.object_box_threshold,
        args.text_threshold,
        args.max_object_candidates,
    )
    selected = select_final_candidates(
        hand_candidates + object_candidates,
        width,
        height,
        args,
        previous_object_box=previous_object_box,
        previous_object_age=previous_object_age,
    )

    with sam3_autocast(args):
        state = sam3_processor.set_image(image)
    final_objects = [refine_with_sam3(image, state, sam3_processor, cand, args) for cand in selected]
    final_objects = [
        obj for obj in final_objects
        if shape_ok(
            obj.box_xyxy,
            width,
            height,
            args.min_hand_area if obj.class_id == HAND_CLASS_ID else args.min_object_area,
            args.max_hand_area if obj.class_id == HAND_CLASS_ID else args.max_object_area,
            args.min_hand_aspect if obj.class_id == HAND_CLASS_ID else args.min_object_aspect,
            args.max_hand_aspect if obj.class_id == HAND_CLASS_ID else args.max_object_aspect,
            args.max_hand_width if obj.class_id == HAND_CLASS_ID else args.max_object_width,
            args.max_hand_height if obj.class_id == HAND_CLASS_ID else args.max_object_height,
        )
    ]

    label_path = args.out_label_dir / f"{image_path.stem}.txt"
    seg_path = args.out_seg_label_dir / f"{image_path.stem}.txt"
    vis_path = args.out_visualized_dir / f"{image_path.stem}{image_path.suffix.lower()}"
    write_detect_label(label_path, final_objects, width, height)
    seg_count = write_segment_label(seg_path, final_objects, width, height, args.segmentation_fallback)
    if args.save_visualized or args.retain_all_frames:
        draw_visualized(image, final_objects, vis_path)

    object_selection_scores = [
        {
            "box_xyxy": list(c.box_xyxy),
            "score": c.score,
            "prompt": c.prompt,
            "temporal_iou": box_iou(c.box_xyxy, previous_object_box) if previous_object_box else 0.0,
            "temporal_compatibility": temporal_compatibility(c, previous_object_box, width, height),
            "temporal_bonus": temporal_object_bonus(
                c, previous_object_box, previous_object_age, width, height, args
            ),
            "total_score": object_selection_score(
                c, hand_candidates, previous_object_box, previous_object_age, width, height, args
            ),
        }
        for c in object_candidates
    ]
    next_object_box = next((obj.box_xyxy for obj in final_objects if obj.class_id != HAND_CLASS_ID), None)

    write_jsonl(args.out_debug_jsonl, {
        "image": image_path.name,
        "previous_object_box": list(previous_object_box) if previous_object_box else None,
        "previous_object_age": previous_object_age,
        "temporal_object_iou_weight": args.temporal_object_iou_weight,
        "hand_candidates": [asdict(c) for c in hand_candidates],
        "object_candidates": [asdict(c) for c in object_candidates],
        "object_selection_scores": object_selection_scores,
        "selected": [asdict(c) for c in selected],
        "final": [asdict(o) for o in final_objects],
    })
    return len(final_objects), seg_count, len(hand_candidates), len(object_candidates), next_object_box


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text("\n".join(f"{k}={v}" for k, v in summary.items()) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_label_dir", required=True)
    parser.add_argument("--out_seg_label_dir", required=True)
    parser.add_argument("--out_visualized_dir", required=True)
    parser.add_argument("--out_debug_jsonl", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--product_class_id", type=int, default=None)
    parser.add_argument("--product_name", default="")
    parser.add_argument("--product_prompts", default="")
    parser.add_argument("--hand_prompts", default="hand|human hand|palm|fingers")
    parser.add_argument("--dino_model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--dino_device", default="cuda")
    parser.add_argument("--sam3_device", default="cuda")
    parser.add_argument("--hand_box_threshold", type=float, default=0.25)
    parser.add_argument("--object_box_threshold", type=float, default=0.22)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--sam3_confidence", type=float, default=0.35)
    parser.add_argument("--sam3_autocast_dtype", choices=["bfloat16", "float16", "none"], default="bfloat16")
    parser.add_argument("--sam3_iou_weight", type=float, default=0.30)
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
    parser.add_argument("--segmentation_fallback", choices=["skip", "box"], default="skip")
    parser.add_argument("--save_visualized", action="store_true", default=True)
    parser.add_argument("--no_save_visualized", dest="save_visualized", action="store_false")
    parser.add_argument("--retain_all_frames", action="store_true", default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    read_hf_tokens_from_env_file(project_root / ".env")

    img_dir = Path(args.img_dir)
    product_folder = img_dir.parent.name
    args.product_class_id = int(args.product_class_id or class_id_from_folder(product_folder))
    args.product_name = args.product_name or strip_folder_timestamp(product_folder)
    args.hand_prompts = split_prompts(args.hand_prompts)
    args.object_prompts = split_prompts(args.product_prompts) or product_prompts_from_folder(product_folder)
    args.out_label_dir = Path(args.out_label_dir)
    args.out_seg_label_dir = Path(args.out_seg_label_dir)
    args.out_visualized_dir = Path(args.out_visualized_dir)
    args.out_debug_jsonl = Path(args.out_debug_jsonl)
    args.summary_path = Path(args.summary_path)

    for directory in [args.out_label_dir, args.out_seg_label_dir, args.out_visualized_dir, args.out_debug_jsonl.parent, args.summary_path.parent]:
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

    print(f"[info] images={len(files)} img_dir={img_dir}")
    print(f"[info] product_class_id={args.product_class_id} product_name={args.product_name}")
    print(f"[info] hand_prompts={args.hand_prompts}")
    print(f"[info] object_prompts={args.object_prompts}")
    print(f"[info] loading Grounding DINO: {args.dino_model}")
    dino_processor, dino_model = load_grounding_dino(args.dino_model, args.dino_device)
    print("[info] loading SAM3 image model")
    sam3_processor = build_sam3_processor(args.sam3_device, args.sam3_confidence)

    started = time.time()
    non_empty = 0
    seg_non_empty = 0
    hand_candidate_count = 0
    object_candidate_count = 0
    previous_object_box: tuple[float, float, float, float] | None = None
    previous_object_age = 0
    previous_frame_number: int | None = None
    for idx, image_path in enumerate(files, start=1):
        current_frame_number = frame_number(image_path)
        if (
            previous_object_box is not None
            and previous_frame_number is not None
            and current_frame_number is not None
            and current_frame_number - previous_frame_number > args.temporal_max_gap
        ):
            previous_object_box = None
            previous_object_age = 0
        if args.verbose or idx == 1 or idx % 100 == 0:
            print(f"[image {idx}/{len(files)}] {image_path.name}")
        det_count, seg_count, hand_count, object_count, detected_object_box = process_image(
            image_path,
            dino_processor,
            dino_model,
            sam3_processor,
            args,
            previous_object_box,
            previous_object_age,
        )
        if detected_object_box is not None:
            previous_object_box = detected_object_box
            previous_object_age = 0
        elif previous_object_box is not None:
            previous_object_age += 1
            if previous_object_age > args.temporal_max_gap:
                previous_object_box = None
                previous_object_age = 0
        previous_frame_number = current_frame_number
        if det_count:
            non_empty += 1
        if seg_count:
            seg_non_empty += 1
        hand_candidate_count += hand_count
        object_candidate_count += object_count

    summary = {
        "backend": "grounding_dino_plus_sam3_image",
        "images": len(files),
        "start_index": args.start_index,
        "detect_non_empty": non_empty,
        "detect_empty": len(files) - non_empty,
        "seg_non_empty": seg_non_empty,
        "seg_empty": len(files) - seg_non_empty,
        "hand_candidates": hand_candidate_count,
        "object_candidates": object_candidate_count,
        "product_class_id": args.product_class_id,
        "product_name": args.product_name,
        "dino_model": args.dino_model,
        "hand_box_threshold": args.hand_box_threshold,
        "object_box_threshold": args.object_box_threshold,
        "sam3_confidence": args.sam3_confidence,
        "temporal_object_iou_weight": args.temporal_object_iou_weight,
        "object_solo_threshold": args.object_solo_threshold,
        "hand_solo_threshold": args.hand_solo_threshold,
        "temporal_decay": args.temporal_decay,
        "temporal_max_gap": args.temporal_max_gap,
        "elapsed_sec": round(time.time() - started, 2),
    }
    write_summary(args.summary_path, summary)
    print(f"[done] {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
