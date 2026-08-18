#!/usr/bin/env python3
"""Prepare and evaluate a product-independent SAM3+DINO benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PACKAGE_TYPES = ("BAG", "BOTTLE", "BOX", "CAN")


def image_files(camera_dir: Path) -> list[Path]:
    direct = sorted(
        path for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if direct:
        return direct
    nested = camera_dir / "images"
    if nested.is_dir():
        return sorted(
            path for path in nested.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    return []


def package_type(folder_name: str) -> str:
    tokens = folder_name.upper().split("_")
    for token in tokens[:3]:
        if token in PACKAGE_TYPES:
            return token
    return "OTHER"


def select_products(source_root: Path, max_products: int) -> list[tuple[Path, str]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(item for item in source_root.iterdir() if item.is_dir()):
        if any((path / camera).is_dir() for camera in ("cam_0", "cam_2")):
            groups[package_type(path.name)].append(path)

    selected: list[tuple[Path, str]] = []
    round_index = 0
    group_names = sorted(groups, key=lambda name: (name == "OTHER", name))
    while len(selected) < max_products:
        added = False
        for group_name in group_names:
            items = groups[group_name]
            if round_index >= len(items) or len(selected) >= max_products:
                continue
            split = "development" if round_index % 2 == 0 else "holdout"
            selected.append((items[round_index], split))
            added = True
        if not added:
            break
        round_index += 1
    return selected


def window_indices(count: int, windows: int, window_size: int) -> list[int]:
    if count <= 0:
        return []
    wanted = min(count, max(1, windows) * max(1, window_size))
    if wanted >= count:
        return list(range(count))
    if windows <= 1:
        centers = [(count - 1) // 2]
    else:
        centers = [round(index * (count - 1) / (windows - 1)) for index in range(windows)]

    indices: list[int] = []
    seen: set[int] = set()
    for center in centers:
        start = max(0, min(count - window_size, center - window_size // 2))
        for index in range(start, min(count, start + window_size)):
            if index not in seen:
                indices.append(index)
                seen.add(index)
    return sorted(indices)


def find_reference(reference_root: Path | None, product: str) -> Path | None:
    if reference_root is None:
        return None
    product_root = reference_root / product
    matches = sorted(product_root.glob("annotations*.json")) if product_root.is_dir() else []
    return matches[0] if matches else None


def prepare(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    reference_root = Path(args.reference_root) if args.reference_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, object]] = []
    selected = select_products(source_root, args.max_products)
    if not selected:
        raise RuntimeError(f"no product directories found: {source_root}")

    for product_dir, split in selected:
        reference = find_reference(reference_root, product_dir.name)
        reference_output = None
        if reference is not None:
            reference_output = output_root / "references" / product_dir.name / reference.name
            reference_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference, reference_output)

        for camera in args.cameras:
            camera_dir = product_dir / camera
            if not camera_dir.is_dir():
                continue
            files = image_files(camera_dir)
            for source_index in window_indices(len(files), args.windows, args.window_size):
                source = files[source_index]
                destination = output_root / "images" / product_dir.name / camera / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                samples.append({
                    "product": product_dir.name,
                    "package_type": package_type(product_dir.name),
                    "split": split,
                    "camera": camera,
                    "file_name": source.name,
                    "source_index": source_index,
                    "source_count": len(files),
                    "relative_image": str(destination.relative_to(output_root)),
                    "reference": str(reference_output.relative_to(output_root)) if reference_output else None,
                })

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": hashlib.sha256(str(source_root).encode()).hexdigest()[:12],
        "selection": {
            "max_products": args.max_products,
            "windows": args.windows,
            "window_size": args.window_size,
            "cameras": args.cameras,
        },
        "products": [
            {"name": path.name, "package_type": package_type(path.name), "split": split}
            for path, split in selected
        ],
        "samples": samples,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    print(f"products={len(selected)} samples={len(samples)}")
    for product in manifest["products"]:
        print(f"{product['split']} {product['package_type']} {product['name']}")
    return 0


def normalize_camera(value: str) -> str:
    match = re.search(r"cam_?(\d+)", value.lower())
    return f"cam_{match.group(1)}" if match else value


def yolo_boxes(path: Path, width: int, height: int) -> dict[str, list[tuple[float, float, float, float]]]:
    result: dict[str, list[tuple[float, float, float, float]]] = {"hand": [], "object": []}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        result["hand" if class_id == 0 else "object"].append((x1, y1, x2, y2))
    return result


def segmentation_classes(path: Path) -> set[str]:
    classes: set[str] = set()
    if not path.exists():
        return classes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        classes.add("hand" if class_id == 0 else "object")
    return classes


def reference_boxes(path: Path | None) -> dict[tuple[str, str], dict[str, list[tuple[float, float, float, float]]]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    category_names = {item["id"]: str(item.get("name", "")).lower() for item in payload.get("categories", [])}
    images = {item["id"]: item for item in payload.get("images", [])}
    result: dict[tuple[str, str], dict[str, list[tuple[float, float, float, float]]]] = defaultdict(
        lambda: {"hand": [], "object": []}
    )
    for annotation in payload.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        if not image:
            continue
        file_name = str(image.get("file_name", ""))
        parts = Path(file_name).parts
        camera = next((normalize_camera(part) for part in parts if re.fullmatch(r"cam_?\d+", part.lower())), "")
        key = (camera, Path(file_name).name)
        x, y, width, height = map(float, annotation.get("bbox", [0, 0, 0, 0]))
        class_name = "hand" if category_names.get(annotation.get("category_id")) == "hand" else "object"
        result[key][class_name].append((x, y, x + width, y + height))
    return dict(result)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float], width: int, height: int) -> float:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot((ax - bx) / max(1, width), (ay - by) / max(1, height))


def greedy_matches(predicted: list, reference: list, threshold: float) -> tuple[int, list[float]]:
    pairs = sorted(
        ((box_iou(pred_box, ref_box), pred_index, ref_index)
         for pred_index, pred_box in enumerate(predicted)
         for ref_index, ref_box in enumerate(reference)),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matched_ious: list[float] = []
    for iou, pred_index, ref_index in pairs:
        if iou < threshold or pred_index in used_pred or ref_index in used_ref:
            continue
        used_pred.add(pred_index)
        used_ref.add(ref_index)
        matched_ious.append(iou)
    return len(matched_ious), matched_ious


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(args: argparse.Namespace) -> int:
    benchmark_root = Path(args.benchmark_root)
    results_root = Path(args.results_root)
    manifest = json.loads((benchmark_root / "manifest.json").read_text(encoding="utf-8"))
    reference_cache: dict[str, dict] = {}
    counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    temporal: dict[tuple[str, str], list[tuple[int, tuple, int, int]]] = defaultdict(list)

    for sample in manifest["samples"]:
        product = sample["product"]
        camera = sample["camera"]
        file_name = sample["file_name"]
        split = sample["split"]
        image_path = benchmark_root / sample["relative_image"]
        with Image.open(image_path) as image:
            width, height = image.size
        predicted = yolo_boxes(results_root / product / camera / "labels" / f"{Path(file_name).stem}.txt", width, height)
        segment = segmentation_classes(
            results_root / product / camera / "labels_seg" / f"{Path(file_name).stem}.txt"
        )

        reference_relative = sample.get("reference")
        if product not in reference_cache:
            reference_cache[product] = reference_boxes(
                benchmark_root / reference_relative if reference_relative else None
            )
        reference = reference_cache[product].get((camera, file_name), {"hand": [], "object": []})

        keys = ("all", split, f"type:{sample['package_type']}", f"product:{product}")
        for key in keys:
            count = counters[key]
            count["frames"] += 1
            has_hand = bool(predicted["hand"])
            has_object = bool(predicted["object"])
            count["pred_hand_frames"] += has_hand
            count["pred_object_frames"] += has_object
            count["pred_both_frames"] += has_hand and has_object
            count["pred_empty_frames"] += not has_hand and not has_object
            count["seg_hand_frames"] += "hand" in segment
            count["seg_object_frames"] += "object" in segment

            for class_name in ("hand", "object"):
                pred_boxes = predicted[class_name]
                ref_boxes = reference[class_name]
                matches, matched_ious = greedy_matches(pred_boxes, ref_boxes, args.iou_threshold)
                count[f"{class_name}_pred"] += len(pred_boxes)
                count[f"{class_name}_ref"] += len(ref_boxes)
                count[f"{class_name}_tp"] += matches
                count[f"{class_name}_iou_sum"] += sum(matched_ious)

        if predicted["object"]:
            temporal[(product, camera)].append(
                (int(sample["source_index"]), predicted["object"][0], width, height)
            )

    for sequence in temporal.values():
        sequence.sort(key=lambda item: item[0])
        for previous, current in zip(sequence, sequence[1:]):
            if current[0] - previous[0] != 1:
                continue
            iou = box_iou(previous[1], current[1])
            distance = center_distance(previous[1], current[1], current[2], current[3])
            for key in ("all",):
                counters[key]["temporal_pairs"] += 1
                counters[key]["temporal_iou_sum"] += iou
                counters[key]["temporal_jump"] += iou < 0.05 and distance > 0.15

    report: dict[str, dict[str, float]] = {}
    for key, raw in sorted(counters.items()):
        frames = raw["frames"]
        item = {name: round(value, 6) for name, value in raw.items()}
        for class_name in ("hand", "object"):
            predicted_count = raw[f"{class_name}_pred"]
            reference_count = raw[f"{class_name}_ref"]
            true_positive = raw[f"{class_name}_tp"]
            precision = safe_ratio(true_positive, predicted_count)
            recall = safe_ratio(true_positive, reference_count)
            item[f"{class_name}_precision_at_iou"] = round(precision, 6)
            item[f"{class_name}_recall_at_iou"] = round(recall, 6)
            item[f"{class_name}_f1_at_iou"] = round(safe_ratio(2 * precision * recall, precision + recall), 6)
            item[f"{class_name}_mean_matched_iou"] = round(
                safe_ratio(raw[f"{class_name}_iou_sum"], true_positive), 6
            )
        item["pred_hand_rate"] = round(safe_ratio(raw["pred_hand_frames"], frames), 6)
        item["pred_object_rate"] = round(safe_ratio(raw["pred_object_frames"], frames), 6)
        item["pred_both_rate"] = round(safe_ratio(raw["pred_both_frames"], frames), 6)
        item["pred_empty_rate"] = round(safe_ratio(raw["pred_empty_frames"], frames), 6)
        item["seg_hand_rate"] = round(safe_ratio(raw["seg_hand_frames"], frames), 6)
        item["seg_object_rate"] = round(safe_ratio(raw["seg_object_frames"], frames), 6)
        item["temporal_mean_iou"] = round(safe_ratio(raw["temporal_iou_sum"], raw["temporal_pairs"]), 6)
        item["temporal_jump_rate"] = round(safe_ratio(raw["temporal_jump"], raw["temporal_pairs"]), 6)
        report[key] = item

    payload = {
        "benchmark": str(benchmark_root),
        "results": str(results_root),
        "iou_threshold": args.iou_threshold,
        "reference_notice": "Pseudo-reference agreement is not human-verified accuracy.",
        "metrics": report,
    }
    output_path = Path(args.output) if args.output else results_root / "benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"report={output_path}")
    for key in ("all", "development", "holdout"):
        if key not in report:
            continue
        item = report[key]
        print(
            key,
            f"frames={int(item['frames'])}",
            f"hand_f1={item['hand_f1_at_iou']:.3f}",
            f"object_f1={item['object_f1_at_iou']:.3f}",
            f"both={item['pred_both_rate']:.3f}",
            f"empty={item['pred_empty_rate']:.3f}",
            f"jump={item['temporal_jump_rate']:.3f}",
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create a diverse, deterministic image benchmark")
    prepare_parser.add_argument("--source-root", required=True)
    prepare_parser.add_argument("--reference-root")
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--max-products", type=int, default=8)
    prepare_parser.add_argument("--windows", type=int, default=3)
    prepare_parser.add_argument("--window-size", type=int, default=2)
    prepare_parser.add_argument("--cameras", nargs="+", default=["cam_0", "cam_2"])
    prepare_parser.set_defaults(func=prepare)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate YOLO labels against common metrics")
    evaluate_parser.add_argument("--benchmark-root", required=True)
    evaluate_parser.add_argument("--results-root", required=True)
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--output")
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
