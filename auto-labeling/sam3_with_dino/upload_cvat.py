#!/usr/bin/env python3
"""Create a CVAT task from a SAM3+DINO COCO result and upload bbox annotations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from dotenv import load_dotenv


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAMERA_DIRS = {
    "cam0": ("cam_0", "cam0"),
    "cam2": ("cam_2", "cam2"),
}


class CvatError(RuntimeError):
    pass


class CvatClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        verify_tls: bool,
        request_timeout: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify_tls
        self.request_timeout = request_timeout
        response = self.session.post(
            self.url("/auth/login"),
            json={"username": username, "password": password},
            timeout=request_timeout,
        )
        self.raise_for_status(response, "CVAT login")
        key = response.json().get("key")
        if not key:
            raise CvatError("CVAT login response did not contain a token")
        self.session.headers["Authorization"] = f"Token {key}"
        self.session.headers["Accept"] = "application/vnd.cvat+json"

    def url(self, path: str) -> str:
        return f"{self.base_url}/api{path}"

    @staticmethod
    def raise_for_status(response: requests.Response, action: str) -> None:
        if response.ok:
            return
        detail = response.text[:1000].replace("\n", " ")
        raise CvatError(f"{action} failed: HTTP {response.status_code}: {detail}")

    def get_json(self, path: str) -> dict[str, Any]:
        response = self.session.get(self.url(path), timeout=self.request_timeout)
        self.raise_for_status(response, f"GET {path}")
        return response.json()

    def create_task(self, name: str, project_id: int, subset: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "project_id": project_id}
        if subset:
            payload["subset"] = subset
        response = self.session.post(
            self.url("/tasks"),
            json=payload,
            timeout=self.request_timeout,
        )
        self.raise_for_status(response, "task creation")
        return response.json()

    def delete_task(self, task_id: int) -> None:
        response = self.session.delete(
            self.url(f"/tasks/{task_id}"),
            timeout=self.request_timeout,
        )
        if response.status_code not in (204, 404):
            self.raise_for_status(response, "incomplete task cleanup")

    def attach_archive(
        self,
        task_id: int,
        archive_path: Path,
        *,
        image_quality: int,
        data_timeout: int,
    ) -> None:
        with archive_path.open("rb") as archive:
            response = self.session.post(
                self.url(f"/tasks/{task_id}/data/"),
                data={
                    "image_quality": str(image_quality),
                    "sorting_method": "natural",
                    "use_zip_chunks": "true",
                },
                files={
                    "client_files[0]": (
                        archive_path.name,
                        archive,
                        "application/zip",
                    )
                },
                timeout=(30, data_timeout),
            )
        self.raise_for_status(response, "task image upload")
        request_id = response.json().get("rq_id")
        if not request_id:
            raise CvatError("task image upload response did not contain rq_id")
        self.wait_for_request(request_id, timeout=data_timeout)

    def wait_for_request(self, request_id: str, *, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload = self.get_json(f"/requests/{request_id}")
            status = payload.get("status")
            if status == "finished":
                return payload
            if status == "failed":
                message = payload.get("message") or payload.get("result") or payload
                raise CvatError(f"CVAT request {request_id} failed: {message}")
            time.sleep(2)
        raise CvatError(f"CVAT request {request_id} timed out after {timeout}s")

    def put_annotations(self, task_id: int, shapes: list[dict[str, Any]]) -> None:
        response = self.session.put(
            self.url(f"/tasks/{task_id}/annotations/"),
            json={"version": 0, "tags": [], "shapes": shapes, "tracks": []},
            timeout=(30, self.request_timeout),
        )
        self.raise_for_status(response, "annotation upload")


def normalize_name(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")


def source_image_for(product_dir: Path, file_name: str) -> Path:
    parts = PurePosixPath(normalize_name(file_name)).parts
    camera = next((part for part in parts if part in CAMERA_DIRS), None)
    if camera is None:
        raise CvatError(f"COCO image has no cam0/cam2 component: {file_name}")
    image_name = parts[-1]
    for source_camera in CAMERA_DIRS[camera]:
        camera_root = product_dir / source_camera
        nested_root = camera_root / "images"
        if nested_root.is_dir():
            camera_root = nested_root
        candidate = camera_root / image_name
        if candidate.is_file():
            return candidate
    raise CvatError(f"source image not found for {file_name}")


def make_image_archive(
    coco: dict[str, Any],
    product_dir: Path,
    destination: Path,
) -> int:
    seen: set[str] = set()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for image in coco.get("images", []):
            file_name = normalize_name(str(image["file_name"]))
            if file_name in seen:
                raise CvatError(f"duplicate COCO image path: {file_name}")
            source = source_image_for(product_dir, file_name)
            if source.suffix.lower() not in IMAGE_EXTS:
                raise CvatError(f"unsupported source image: {source}")
            archive.write(source, arcname=file_name)
            seen.add(file_name)
    if not seen:
        raise CvatError("COCO result contains no images")
    return len(seen)


def resolve_frame(
    file_name: str,
    frames_by_name: dict[str, list[int]],
) -> int:
    normalized = normalize_name(file_name)
    exact = frames_by_name.get(normalized, [])
    if len(exact) == 1:
        return exact[0]
    suffix_matches = {
        frame
        for name, frame_numbers in frames_by_name.items()
        if name.endswith(f"/{normalized}") or normalized.endswith(f"/{name}")
        for frame in frame_numbers
    }
    if len(suffix_matches) == 1:
        return suffix_matches.pop()
    raise CvatError(f"unable to map COCO image to one CVAT frame: {file_name}")


def build_shapes(
    coco: dict[str, Any],
    task_meta: dict[str, Any],
    project: dict[str, Any],
) -> list[dict[str, Any]]:
    project_labels = {
        str(label["name"]): int(label["id"])
        for label in project.get("labels", [])
    }
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    missing_labels = sorted(set(categories.values()) - set(project_labels))
    if missing_labels:
        raise CvatError(f"project is missing labels: {missing_labels}")

    task_frames = task_meta.get("frames", [])
    frames_by_name: dict[str, list[int]] = {}
    for index, frame in enumerate(task_frames):
        frames_by_name.setdefault(normalize_name(str(frame["name"])), []).append(index)

    coco_images = list(coco.get("images", []))
    images = {
        int(image["id"]): image
        for image in coco_images
    }
    image_frames: dict[int, int] = {}
    for position, image in enumerate(coco_images):
        image_id = int(image["id"])
        try:
            image_frames[image_id] = resolve_frame(
                str(image["file_name"]), frames_by_name
            )
        except CvatError:
            if len(task_frames) != len(coco_images):
                raise
            image_frames[image_id] = position

    shapes: list[dict[str, Any]] = []
    for annotation in coco.get("annotations", []):
        image = images.get(int(annotation["image_id"]))
        if image is None:
            raise CvatError(f"annotation references missing image: {annotation}")
        category_name = categories.get(int(annotation["category_id"]))
        if category_name is None:
            raise CvatError(f"annotation references missing category: {annotation}")
        x, y, width, height = (float(value) for value in annotation["bbox"])
        max_width = float(image["width"])
        max_height = float(image["height"])
        x1 = max(0.0, min(max_width, x))
        y1 = max(0.0, min(max_height, y))
        x2 = max(0.0, min(max_width, x + width))
        y2 = max(0.0, min(max_height, y + height))
        if x2 <= x1 or y2 <= y1:
            continue
        shapes.append(
            {
                "type": "rectangle",
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "points": [x1, y1, x2, y2],
                "label_id": project_labels[category_name],
                "group": 0,
                "source": "auto",
                "frame": image_frames[int(image["id"])],
                "attributes": [],
            }
        )
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("CVAT_BASE_URL", "https://aipnt.atcrk.co.kr/cvat"))
    parser.add_argument("--username", default=os.environ.get("CVAT_USERNAME", ""))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--project-id", type=int, default=int(os.environ.get("CVAT_PROJECT_ID", "3")))
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--subset", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--image-quality", type=int, default=95)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--insecure", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    args = parse_args()
    args.username = args.username or os.environ.get("CVAT_USERNAME", "")
    args.password = args.password or os.environ.get("CVAT_PASSWORD", "")
    if not args.username or not args.password:
        print("[error] CVAT_USERNAME and CVAT_PASSWORD are required", file=sys.stderr)
        return 2

    product_dir = Path(args.product_dir).resolve()
    annotation_path = Path(args.annotation_json).resolve()
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    client = CvatClient(
        args.base_url,
        args.username,
        args.password,
        verify_tls=not args.insecure,
        request_timeout=args.request_timeout,
    )
    project = client.get_json(f"/projects/{args.project_id}")
    project_labels = client.get_json(
        f"/labels?project_id={args.project_id}&page_size=1000"
    )
    project["labels"] = project_labels.get("results", project_labels)
    task_id: int | None = None
    try:
        task = client.create_task(args.task_name, args.project_id, args.subset)
        task_id = int(task["id"])
        with tempfile.TemporaryDirectory(prefix="cvat_upload.") as temp_dir:
            archive_path = Path(temp_dir) / f"{product_dir.name}.zip"
            image_count = make_image_archive(coco, product_dir, archive_path)
            print(f"[cvat] task={task_id} archive_images={image_count}")
            client.attach_archive(
                task_id,
                archive_path,
                image_quality=args.image_quality,
                data_timeout=args.request_timeout,
            )
        task_meta = client.get_json(f"/tasks/{task_id}/data/meta")
        shapes = build_shapes(coco, task_meta, project)
        client.put_annotations(task_id, shapes)
        uploaded = client.get_json(f"/tasks/{task_id}/annotations/")
        if len(uploaded.get("shapes", [])) != len(shapes):
            raise CvatError(
                f"annotation verification mismatch: sent={len(shapes)} "
                f"stored={len(uploaded.get('shapes', []))}"
            )
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "task_name": args.task_name,
                    "project_id": args.project_id,
                    "images": len(task_meta.get("frames", [])),
                    "shapes": len(shapes),
                    "url": f"{args.base_url}/tasks/{task_id}",
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception:
        if task_id is not None:
            client.delete_task(task_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
