#!/usr/bin/env python3
"""Build a CVAT-compatible COCO ZIP from YOLO hand and product labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def source_image(image_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image not found for frame {stem} in {image_dir}")


def yolo_boxes(path: Path, class_id: int) -> list[tuple[float, float, float, float]]:
    if not path.is_file():
        return []
    result = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = raw_line.split()
        if len(fields) < 5 or int(float(fields[0])) != class_id:
            continue
        result.append(tuple(map(float, fields[1:5])))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--frame-list-dir", type=Path, required=True)
    parser.add_argument("--hand-label-dir", type=Path, required=True)
    parser.add_argument("--product-label-dir", type=Path, required=True)
    parser.add_argument("--product-class-id", type=int, required=True)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    frame_names = sorted(path.stem for path in args.frame_list_dir.glob("*.txt"))
    if not frame_names:
        parser.error(f"no frame list labels in {args.frame_list_dir}")

    images = []
    annotations = []
    image_paths: list[Path] = []
    annotation_id = 1
    for image_id, frame_name in enumerate(frame_names, 1):
        image_path = source_image(args.image_dir, frame_name)
        with Image.open(image_path) as image:
            width, height = int(image.width), int(image.height)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        image_paths.append(image_path)

        sources = (
            (args.hand_label_dir / f"{frame_name}.txt", 0, 1),
            (
                args.product_label_dir / f"{frame_name}.txt",
                args.product_class_id,
                2,
            ),
        )
        for label_path, source_class_id, category_id in sources:
            for center_x, center_y, width_n, height_n in yolo_boxes(
                label_path, source_class_id
            ):
                box_width = max(0.0, width_n * width)
                box_height = max(0.0, height_n * height)
                x = max(0.0, center_x * width - box_width / 2.0)
                y = max(0.0, center_y * height - box_height / 2.0)
                box_width = min(box_width, width - x)
                box_height = min(box_height, height - y)
                if box_width <= 0.0 or box_height <= 0.0:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, box_width, box_height],
                        "area": box_width * box_height,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    payload = {
        "info": {"description": args.description, "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "hand", "supercategory": "none"},
            {"id": 2, "name": args.product_name, "supercategory": "none"},
        ],
    }

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with zipfile.ZipFile(args.output_zip, "w", allowZip64=True) as archive:
        archive.writestr(
            "annotations/instances_default.json",
            json_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for image_path in image_paths:
            archive.write(
                image_path,
                f"images/default/{image_path.name}",
                compress_type=zipfile.ZIP_STORED,
            )

    with zipfile.ZipFile(args.output_zip) as archive:
        members = set(archive.namelist())
        loaded = json.loads(archive.read("annotations/instances_default.json"))
        missing = [
            item["file_name"]
            for item in loaded["images"]
            if f"images/default/{item['file_name']}" not in members
        ]
        if missing:
            raise RuntimeError(f"COCO JSON image paths missing from archive: {missing[:5]}")
        if len(loaded["images"]) != len(frame_names):
            raise RuntimeError("COCO image count changed during packaging")

    digest = hashlib.sha256(args.output_zip.read_bytes()).hexdigest()
    hand_annotations = sum(item["category_id"] == 1 for item in annotations)
    product_annotations = sum(item["category_id"] == 2 for item in annotations)
    print(
        json.dumps(
            {
                "output": str(args.output_zip),
                "sha256": digest,
                "images": len(images),
                "hand_annotations": hand_annotations,
                "product_annotations": product_annotations,
                "bytes": args.output_zip.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
