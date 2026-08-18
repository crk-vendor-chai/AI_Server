"""Upload existing SAM3+DINO outputs as complete MinIO web-review packages."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath

from annotation_job import upload_review_package


def _read_package_identity(annotations_path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("COCO JSON has no images")

    roots: set[str] = set()
    camera_numbers: set[int] = set()
    for image in images:
        file_name = str(image.get("file_name") or "") if isinstance(image, dict) else ""
        parts = PurePosixPath(file_name).parts
        if len(parts) < 3:
            raise ValueError(f"invalid COCO image path: {file_name}")

        match = re.fullmatch(r"cam(\d+)", parts[-2])
        if match is None:
            raise ValueError(f"invalid camera folder: {file_name}")
        camera_numbers.add(int(match.group(1)))
        roots.add("/".join(parts[:-2]))

    if len(roots) != 1:
        raise ValueError(f"multiple package roots in COCO JSON: {sorted(roots)}")

    root_folder = roots.pop()
    try:
        brunch_name, train_product_idx, product_idx = root_folder.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"invalid package root: {root_folder}") from exc

    product_info = {
        "brunch_name": brunch_name,
        "train_product_idx": train_product_idx,
        "product_idx": product_idx,
    }
    cams = [f"cam_{number}" for number in sorted(camera_numbers)]
    return product_info, cams


def sync_product(annotations_path: Path, img_root: Path) -> bool:
    product_root = annotations_path.parent.parent
    local_image_root = img_root / product_root.name
    try:
        product_info, cams = _read_package_identity(annotations_path)
    except Exception as exc:
        print(f"[review_sync] invalid annotation JSON {annotations_path}: {exc}", file=sys.stderr, flush=True)
        return False

    print(
        f"[review_sync] upload start: product={product_root.name}, cams={cams}, "
        f"json={annotations_path.name}",
        flush=True,
    )
    ok = upload_review_package(product_info, annotations_path, local_image_root, cams)
    print(f"[review_sync] upload {'complete' if ok else 'failed'}: {product_root.name}", flush=True)
    return ok


def _annotation_files(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("*/out/annotations_*.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--img-root",
        type=Path,
        default=Path(os.environ.get("ANNOTATION_BASE_DIR", "./img_download")),
    )
    parser.add_argument("--expected-products", type=int, default=0)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--retry-sec", type=float, default=30.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    args = parser.parse_args()

    if not args.run_root.is_dir() or not args.img_root.is_dir():
        print("[review_sync] run root or image root not found", file=sys.stderr)
        return 2

    completed: set[Path] = set()
    last_attempt: dict[Path, float] = {}
    deadline = time.monotonic() + args.timeout_sec if args.timeout_sec > 0 else None
    last_status: tuple[int, int] | None = None

    while True:
        now = time.monotonic()
        files = _annotation_files(args.run_root)
        for path in files:
            if path in completed or now - last_attempt.get(path, 0.0) < args.retry_sec:
                continue
            last_attempt[path] = now
            if sync_product(path, args.img_root):
                completed.add(path)

        status = (len(completed), len(files))
        if status != last_status:
            expected = args.expected_products or len(files)
            print(
                f"[review_sync] progress: uploaded={len(completed)}, "
                f"discovered={len(files)}, expected={expected}",
                flush=True,
            )
            last_status = status

        if args.expected_products <= 0:
            return 0 if len(completed) == len(files) else 1
        if len(completed) >= args.expected_products:
            return 0
        if deadline is not None and now >= deadline:
            print("[review_sync] timeout", file=sys.stderr, flush=True)
            return 1
        time.sleep(max(0.5, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
