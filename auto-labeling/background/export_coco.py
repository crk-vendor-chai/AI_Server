#!/usr/bin/env python3
"""Export hand-only YOLO detection labels as one COCO bbox JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def image_files(camera_dir: Path) -> list[Path]:
    return sorted(
        path for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def parse_detect(path: Path, width: int, height: int) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(float(parts[0])) != 0:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        box_w, box_h = bw * width, bh * height
        rows.append({
            "bbox": [cx * width - box_w / 2, cy * height - box_h / 2, box_w, box_h],
        })
    return rows




def build_payload(final_root: Path, work_root: Path, cameras: list[str]) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 0
    annotation_id = 0

    for camera in cameras:
        camera_dir = final_root / camera
        label_dir = work_root / camera / "labels"
        for image_path in image_files(camera_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise RuntimeError(f"missing detection label for {camera}/{image_path.name}")

            with Image.open(image_path) as image:
                width, height = image.size
            images.append({
                "id": image_id,
                "file_name": f"{final_root.name}/{camera}/{image_path.name}",
                "width": width,
                "height": height,
            })

            for detection in parse_detect(label_path, width, height):
                bbox = detection["bbox"]
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                })
                annotation_id += 1
            image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "hand", "supercategory": "person"}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=["cam0", "cam4", "cam6"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = build_payload(args.final_root, args.work_root, args.cameras)
    output = args.output or args.final_root / "annotations_hand_background.json"
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
