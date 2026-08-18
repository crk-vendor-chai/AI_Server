#!/usr/bin/env python3
"""Export final detection labels to the existing web/CVAT COCO JSON format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def strip_foldername_timestamp(foldername: str) -> str:
    name = re.sub(r"_\d{8}_\d{6}$", "", str(foldername or "").strip())
    return re.sub(r"^\d+_", "", name)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return slug.strip("_") or "dataset"


def normalize_cam_upload_folder(cam_name: str) -> str:
    match = re.search(r"cam_(\d+)", str(cam_name or ""))
    if match:
        return f"cam_{match.group(1)}"
    return re.sub(r"[^0-9A-Za-z_]+", "", str(cam_name or "")) or "cam"


def class_id_from_folder(folder_name: str) -> int:
    try:
        return int(folder_name.split("_", 1)[0])
    except Exception:
        return 1


def find_source_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def build_root_folder(args: argparse.Namespace) -> str:
    if args.root_folder:
        return args.root_folder.strip("/")
    parts = [args.brunch_name, str(args.train_product_idx or ""), args.product_idx]
    return "_".join(part.strip("_") for part in parts if part).strip("_")


def build_payload(args: argparse.Namespace) -> dict:
    results_root = Path(args.results_root)
    image_root = Path(args.image_root)
    product_folder = image_root.name
    product_source_id = int(args.product_class_id or class_id_from_folder(product_folder))
    product_name = args.product_name or strip_foldername_timestamp(product_folder)
    root_folder = build_root_folder(args)

    source_to_category = {0: 1, product_source_id: 2}
    categories = [
        {"id": 1, "name": "hand", "supercategory": "none"},
        {"id": 2, "name": product_name, "supercategory": "none"},
    ]

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 0
    annotation_id = 0

    for cam_name in args.cams:
        label_dir = results_root / cam_name / "labels"
        source_image_dir = image_root / cam_name
        nested_image_dir = source_image_dir / "images"
        if nested_image_dir.is_dir():
            source_image_dir = nested_image_dir
        if not label_dir.is_dir():
            raise FileNotFoundError(f"missing label directory: {label_dir}")
        if not source_image_dir.is_dir():
            raise FileNotFoundError(f"missing image directory: {source_image_dir}")

        image_prefix = "/".join(
            part.strip("/")
            for part in (root_folder, normalize_cam_upload_folder(cam_name))
            if part and part.strip("/")
        )

        for txt_path in sorted(label_dir.glob("*.txt")):
            source_image = find_source_image(source_image_dir, txt_path.stem)
            if source_image is None:
                raise FileNotFoundError(f"source image not found for {txt_path}")
            width, height = image_size(source_image)
            images.append({
                "id": image_id,
                "file_name": f"{image_prefix}/{source_image.name}" if image_prefix else source_image.name,
                "width": width,
                "height": height,
            })
            for line in txt_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    source_class_id = int(float(parts[0]))
                    xc, yc, bw_n, bh_n = map(float, parts[1:])
                except Exception:
                    continue
                category_id = source_to_category.get(source_class_id)
                if category_id is None:
                    continue
                box_w = bw_n * width
                box_h = bh_n * height
                x = (xc * width) - (box_w / 2)
                y = (yc * height) - (box_h / 2)
                x = max(0.0, min(float(width), x))
                y = max(0.0, min(float(height), y))
                box_w = max(0.0, min(box_w, float(width) - x))
                box_h = max(0.0, min(box_h, float(height) - y))
                if box_w <= 0.0 or box_h <= 0.0:
                    continue
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                })
                annotation_id += 1
            image_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cams", nargs="+", default=["cam_0", "cam_2"])
    parser.add_argument("--product-class-id", type=int)
    parser.add_argument("--product-name")
    parser.add_argument("--root-folder")
    parser.add_argument("--brunch-name", default="DI17798460900133031_C")
    parser.add_argument("--train-product-idx", default="")
    parser.add_argument("--product-idx", default="")
    args = parser.parse_args()

    payload = build_payload(args)
    output = Path(args.output) if args.output else Path(args.results_root) / "out" / f"annotations_{slugify(payload['categories'][1]['name'])}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=4) + "\n", encoding="utf-8")
    alias = output.parent / "annotation.json"
    if alias != output:
        alias.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"output={output}")
    print(f"alias={alias}")
    print(f"images={len(payload['images'])}")
    print(f"annotations={len(payload['annotations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
