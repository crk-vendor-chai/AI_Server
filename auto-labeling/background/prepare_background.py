#!/usr/bin/env python3
"""Download a background ZIP from MinIO and normalize camera folders."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from minio import Minio


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_object_path(value: str) -> tuple[str, str]:
    normalized = value.strip()
    if normalized.startswith("s3://"):
        normalized = normalized[5:]
    normalized = normalized.strip("/")
    bucket, object_name = normalized.split("/", 1)
    if not bucket or not object_name:
        raise ValueError(f"invalid MinIO object path: {value}")
    return bucket, object_name


def minio_client() -> Minio:
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY is required")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def normalize_camera(source: str) -> str:
    if not source.startswith("cam_") or not source[4:].isdigit():
        raise ValueError(f"unsupported camera folder: {source}")
    return f"cam{int(source[4:])}"


def parse_camera_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        source, destination = value.split("=", 1)
        if not destination.startswith("cam") or not destination[3:].isdigit():
            raise ValueError(f"unsupported output camera: {destination}")
        result[source] = destination
    return result


def safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe ZIP member: {name}")
    return path.parts


def prepare(
    minio_object: str,
    output_root: Path,
    manifest_path: Path,
    camera_map: dict[str, str],
    mapped_only: bool = False,
) -> dict:
    bucket, object_name = parse_object_path(minio_object)
    client = minio_client()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    counts: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="background_zip.") as temp_dir:
        zip_path = Path(temp_dir) / Path(object_name).name
        client.fget_object(bucket, object_name, str(zip_path))

        with ZipFile(zip_path) as archive:
            members: list[tuple[object, tuple[str, ...]]] = []
            for info in archive.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_EXTS:
                    continue
                parts = safe_parts(info.filename)
                if len(parts) == 3:
                    clip_name, source_camera, source_name = parts
                elif len(parts) == 4 and parts[1] == "images":
                    clip_name, _, source_camera, source_name = parts
                else:
                    raise ValueError(
                        "expected <clip>/<camera>/<image> or "
                        f"<clip>/images/<camera>/<image>: {info.filename}"
                    )
                members.append((info, (clip_name, source_camera, source_name)))

            clips = sorted({parts[0] for _, parts in members})
            clip_indices = {clip: index for index, clip in enumerate(clips, start=1)}
            frame_indices: dict[tuple[str, str], int] = {}
            for info, parts in sorted(members, key=lambda item: item[1]):
                clip_name, source_camera, source_name = parts
                if mapped_only and source_camera not in camera_map:
                    continue
                destination_camera = camera_map.get(source_camera, normalize_camera(source_camera))
                key = (clip_name, source_camera)
                frame_indices[key] = frame_indices.get(key, 0) + 1
                destination_name = (
                    f"{clip_indices[clip_name]}_{frame_indices[key]}"
                    f"{Path(source_name).suffix.lower()}"
                )
                destination = output_root / destination_camera / destination_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file() or destination.stat().st_size != info.file_size:
                    with archive.open(info) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)

                counts[destination_camera] = counts.get(destination_camera, 0) + 1
                records.append({
                    "output_camera": destination_camera,
                    "output_name": destination_name,
                    "source_camera": source_camera,
                    "source_path": info.filename,
                    "clip": clip_name,
                })

    payload = {
        "source": f"s3://{bucket}/{object_name}",
        "camera_map": camera_map,
        "mapped_only": mapped_only,
        "cameras": sorted(counts),
        "counts": counts,
        "images": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minio-object", default="chaiimage/etc/background/20260722_12.zip")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--camera-map",
        nargs="+",
        default=[],
    )
    parser.add_argument("--mapped-only", action="store_true")
    args = parser.parse_args()

    camera_map = parse_camera_map(args.camera_map)
    if args.mapped_only and not camera_map:
        raise ValueError("--mapped-only requires at least one --camera-map entry")

    payload = prepare(
        args.minio_object,
        args.output_root,
        args.manifest,
        camera_map,
        args.mapped_only,
    )
    print(json.dumps({key: payload[key] for key in ("source", "cameras", "camera_map", "counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
