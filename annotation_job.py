"""
자동 어노테이션 job — `main.py` product/created (status=2, isNew=1) 시 호출.

흐름:
  1) MinIO `ProductsList.folderpath` → 로컬 `ANNOTATION_BASE_DIR/{foldername}`
  2) Grounding DINO로 손/상품 bbox 탐지
  3) SAM3 image model로 bbox와 segmentation mask 보정
  4) 모든 프레임의 YOLO 라벨과 COCO `annotations_*.json` 생성
  5) 원본 이미지와 COCO JSON을 MinIO 웹 검수용 상품 폴더에 업로드

웹 검수 업로드 구조:
  NewAnnotation/{brunch}_{train}_{product}/
    cam_0/...
    cam_2/...
    annotations_<product>.json

웹/CVAT 검수 후 확정 라벨은 `productAnnotation/`에 올라가며, segment_job이 사용합니다.

환경 변수:
- ANNOTATION_BASE_DIR (기본 `./img_download`)
- SAM3_DINO_RUNNER, SAM3_DINO_RUNS_DIR, SAM3_DINO_* threshold
- MINIO_*
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_ANNOTATION_BASE_DIR = _PROJECT_ROOT / "img_download"
_DEFAULT_SAM3_DINO_RUNNER = _PROJECT_ROOT / "auto-labeling" / "sam3_with_dino" / "run_product.py"
_DEFAULT_SAM3_DINO_RUNS_DIR = _PROJECT_ROOT / "runs" / "sam3_with_dino"

try:
    from .job_log import run_shell_job
    from .request_log import get_request_context
except ImportError:
    from job_log import run_shell_job
    from request_log import get_request_context


_CLASS_COLOR_PALETTE = (
    "#F08CB3",
    "#4ce6df",
    "#6CC8FF",
    "#FFB86B",
    "#9FE870",
    "#FF8C94",
    "#9B8CFF",
    "#FFD166",
)


def _parse_minio_folderpath(folderpath: str) -> Tuple[str, str]:
    """
    ProductsList.folderpath -> (bucket, prefix) 분리.
    예: "/chaiimage/productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426/"
        -> ("chaiimage", "productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426")
       "s3://chaiimage/productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426/"
        -> ("chaiimage", "productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426")
    """
    normalized = folderpath.strip()
    if normalized.startswith("s3://"):
        normalized = normalized[len("s3://"):]
    normalized = normalized.strip("/")
    parts = normalized.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, prefix


def _strip_foldername_timestamp(foldername: str) -> str:
    name = re.sub(r"_\d{8}_\d{6}$", "", str(foldername or "").strip())
    return re.sub(r"^\d+_", "", name)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return slug.strip("_") or "dataset"


def _normalize_cam_upload_folder(cam_name: str) -> str:
    match = re.search(r"cam_(\d+)", str(cam_name or ""))
    if match:
        return f"cam_{match.group(1)}"
    return re.sub(r"[^0-9A-Za-z_]+", "", str(cam_name or "")) or "cam"


def _pick_class_color(source_id: int) -> str:
    return _CLASS_COLOR_PALETTE[source_id % len(_CLASS_COLOR_PALETTE)]


def _build_label_classes(product_info: dict) -> list[dict]:
    classes = [{
        "source_id": 0,
        "name": "hand",
        "color": _CLASS_COLOR_PALETTE[0],
    }]

    raw_classes = product_info.get("label_classes")
    if isinstance(raw_classes, list) and raw_classes:
        seen_ids = {0}
        for item in raw_classes:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id", item.get("train_product_idx", item.get("id")))
            try:
                source_id = int(source_id)
            except Exception:
                continue
            if source_id in seen_ids:
                continue
            name = (
                item.get("name")
                or item.get("product_eng_name")
                or item.get("productEngName")
                or _strip_foldername_timestamp(item.get("foldername") or item.get("folderName") or "")
            )
            name = str(name or "").strip()
            if not name:
                continue
            classes.append({
                "source_id": source_id,
                "name": name,
                "color": str(item.get("color") or _pick_class_color(source_id)),
            })
            seen_ids.add(source_id)
        if len(classes) > 1:
            return classes

    train_product_idx = product_info.get("train_product_idx")
    try:
        source_id = int(train_product_idx)
    except Exception:
        return classes

    product_name = (
        product_info.get("product_eng_name")
        or product_info.get("productEngName")
        or _strip_foldername_timestamp(product_info.get("foldername") or "")
    )
    product_name = str(product_name or "").strip()
    if product_name:
        classes.append({
            "source_id": source_id,
            "name": product_name,
            "color": _pick_class_color(source_id),
        })
    return classes


def _load_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return int(img.width), int(img.height)
    except Exception:
        pass

    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is not None:
            height, width = img.shape[:2]
            return int(width), int(height)
    except Exception:
        pass

    raise RuntimeError(f"이미지 크기 조회 실패: {image_path}")


def _resolve_source_image_dir(local_image_root: Path, cam_name: str) -> Path:
    cam_root = local_image_root / cam_name
    images_dir = cam_root / "images"
    return images_dir if images_dir.is_dir() else cam_root


def _find_source_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _build_annotations_payload(
    results_root: Path,
    local_image_root: Path,
    class_defs: list[dict],
    root_folder: str = "",
) -> dict:
    source_to_category = {
        cls["source_id"]: idx + 1
        for idx, cls in enumerate(class_defs)
    }
    categories = [
        {
            "id": idx + 1,
            "name": cls["name"],
            "supercategory": "none",
        }
        for idx, cls in enumerate(class_defs)
    ]

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    for cam_dir in sorted(results_root.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("cam_"):
            continue
        label_dir = cam_dir / "labels"
        if not label_dir.is_dir():
            continue

        image_dir = _resolve_source_image_dir(local_image_root, cam_dir.name)
        cam_upload_folder = _normalize_cam_upload_folder(cam_dir.name)
        image_prefix = "/".join(
            part.strip("/")
            for part in (root_folder, cam_upload_folder)
            if part and part.strip("/")
        )

        for txt_path in sorted(label_dir.glob("*.txt")):
            source_image = _find_source_image(image_dir, txt_path.stem)
            if source_image is None:
                print(f"[annotation_job] source image not found for {txt_path}", file=sys.stderr)
                continue

            width, height = _load_image_size(source_image)
            images.append({
                "id": image_id,
                "file_name": f"{image_prefix}/{source_image.name}" if image_prefix else source_image.name,
                "width": width,
                "height": height,
            })

            for line in txt_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    source_class_id = int(float(parts[0]))
                    xc, yc, bw_n, bh_n = map(float, parts[1:5])
                except Exception:
                    continue

                category_id = source_to_category.get(source_class_id)
                if category_id is None:
                    continue

                box_w = bw_n * width
                box_h = bh_n * height
                x = (xc * width) - (box_w / 2)
                y = (yc * height) - (box_h / 2)
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

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=4) + "\n", encoding="utf-8")
    return path


def _find_sam3_dino_annotations(product_root: Path) -> Optional[Path]:
    out_dir = product_root / "out"
    if not out_dir.is_dir():
        print(f"[annotation_job] SAM3+DINO output folder missing: {out_dir}", file=sys.stderr)
        return None
    candidates = sorted(out_dir.glob("annotations_*.json"))
    if candidates:
        return candidates[0]
    alias = out_dir / "annotation.json"
    if alias.is_file():
        return alias
    print(f"[annotation_job] SAM3+DINO annotation JSON missing: {out_dir}", file=sys.stderr)
    return None


def _safe_run_component(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "").strip()).strip("_") or "unknown"


def _camera_has_images(cam_dir: Path) -> bool:
    for root in (cam_dir, cam_dir / "images"):
        if not root.is_dir():
            continue
        if any(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in root.iterdir()):
            return True
    return False


def _sam3_dino_run_name(division_idx: str) -> str:
    ctx = get_request_context()
    request_id = ""
    if ctx is not None:
        request_id = ctx.if_sysid or ctx.request_id
    if not request_id:
        request_id = f"{division_idx}_{time.strftime('%Y%m%d_%H%M%S')}"
    return f"api_{_safe_run_component(division_idx)}_{_safe_run_component(request_id)}"

_IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _create_minio_client():
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[annotation_job] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        return None

    try:
        from minio import Minio
    except ImportError:
        print("[annotation_job] minio 라이브러리 없음: pip install minio", file=sys.stderr)
        return None

    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    except Exception as e:
        print(f"[annotation_job] MinIO client 생성 실패: {e}", file=sys.stderr)
        return None


def _build_review_image_manifest(
    annotations_path: Path,
    local_image_root: Path,
    root_folder: str,
) -> Optional[list[tuple[str, Path]]]:
    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[annotation_job] annotation JSON 읽기 실패: {exc}", file=sys.stderr)
        return None

    images = payload.get("images")
    if not isinstance(images, list) or not images:
        print("[annotation_job] annotation JSON에 images가 없습니다", file=sys.stderr)
        return None

    expected_prefix = tuple(PurePosixPath(root_folder).parts)
    manifest: list[tuple[str, Path]] = []
    seen: set[tuple[str, str]] = set()

    for image in images:
        file_name = str(image.get("file_name") or "") if isinstance(image, dict) else ""
        parts = PurePosixPath(file_name).parts
        if len(parts) < 2:
            print(f"[annotation_job] 잘못된 COCO image file_name: {file_name}", file=sys.stderr)
            return None

        relative_prefix = tuple(parts[:-2])
        cam_folder, image_name = parts[-2], parts[-1]
        if expected_prefix and relative_prefix != expected_prefix:
            print(
                f"[annotation_job] COCO image root 불일치: expected={root_folder}, actual={file_name}",
                file=sys.stderr,
            )
            return None

        cam_match = re.fullmatch(r"cam_(\d+)", cam_folder)
        if cam_match is None or Path(image_name).name != image_name:
            print(f"[annotation_job] 잘못된 COCO image 경로: {file_name}", file=sys.stderr)
            return None

        key = (cam_folder, image_name)
        if key in seen:
            print(f"[annotation_job] COCO image 경로 중복: {file_name}", file=sys.stderr)
            return None
        seen.add(key)

        source_dir = _resolve_source_image_dir(local_image_root, f"cam_{cam_match.group(1)}")
        source_image = source_dir / image_name
        if not source_image.is_file():
            source_image = _find_source_image(source_dir, Path(image_name).stem)
        if source_image is None or not source_image.is_file():
            print(f"[annotation_job] 웹 검수 원본 이미지 누락: {file_name}", file=sys.stderr)
            return None
        manifest.append((cam_folder, source_image))

    return manifest


def upload_review_package(
    product_info: dict,
    annotations_path: Path,
    local_image_root: Path,
    cams: list[str],
) -> bool:
    bucket = os.environ.get("MINIO_BUCKET", "chaiimage").strip() or "chaiimage"
    base_prefix = os.environ.get("MINIO_BASE_PREFIX", "NewAnnotation").strip().strip("/") or "NewAnnotation"
    brunch_name = str(product_info.get("brunch_name") or "").strip()
    train_product_idx = str(product_info.get("train_product_idx") or "").strip()
    product_idx = str(product_info.get("product_idx") or "").strip()
    if not (brunch_name and train_product_idx and product_idx):
        print("[annotation_job] 웹 검수 업로드 skip: MinIO root folder 정보 부족", file=sys.stderr)
        return False

    root_folder = f"{brunch_name}_{train_product_idx}_{product_idx}"
    manifest = _build_review_image_manifest(annotations_path, local_image_root, root_folder)
    if manifest is None:
        return False

    expected_cams = {_normalize_cam_upload_folder(cam) for cam in cams}
    manifest_cams = {cam_folder for cam_folder, _ in manifest}
    missing_cams = sorted(expected_cams - manifest_cams)
    if missing_cams:
        print(
            f"[annotation_job] COCO JSON에 카메라 이미지 누락: {', '.join(missing_cams)}",
            file=sys.stderr,
        )
        return False

    client = _create_minio_client()
    if client is None:
        return False

    try:
        workers = max(1, int(os.environ.get("ANNOTATION_MINIO_UPLOAD_WORKERS", "4")))
    except ValueError:
        workers = 4

    uploaded_by_cam = {cam: 0 for cam in sorted(manifest_cams)}

    def _upload_image(item: tuple[str, Path]) -> str:
        cam_folder, image_path = item
        object_name = f"{base_prefix}/{root_folder}/{cam_folder}/{image_path.name}"
        content_type = _IMAGE_CONTENT_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
        client.fput_object(bucket, object_name, str(image_path), content_type=content_type)
        return cam_folder

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(manifest))) as executor:
        future_to_item = {executor.submit(_upload_image, item): item for item in manifest}
        for future in as_completed(future_to_item):
            cam_folder, image_path = future_to_item[future]
            try:
                uploaded_by_cam[future.result()] += 1
            except Exception as exc:
                failures.append(f"{cam_folder}/{image_path.name}: {exc}")

    if failures:
        for failure in failures[:10]:
            print(f"[annotation_job] 웹 검수 이미지 업로드 실패: {failure}", file=sys.stderr)
        if len(failures) > 10:
            print(f"[annotation_job] 추가 업로드 실패 {len(failures) - 10}건", file=sys.stderr)
        return False

    counts = ", ".join(f"{cam}={count}" for cam, count in sorted(uploaded_by_cam.items()))
    print(
        f"[annotation_job] 웹 검수 원본 이미지 업로드 완료: "
        f"{bucket}/{base_prefix}/{root_folder} ({counts}, total={len(manifest)})"
    )

    annotations_object = f"{base_prefix}/{root_folder}/{annotations_path.name}"
    try:
        client.fput_object(
            bucket,
            annotations_object,
            str(annotations_path),
            content_type="application/json",
        )
        print(f"[annotation_job] 웹 검수 COCO JSON 업로드 완료: {bucket}/{annotations_object}")
        return True
    except Exception as exc:
        print(f"[annotation_job] COCO JSON 업로드 실패 ({annotations_object}): {exc}", file=sys.stderr)
        return False


def download_images_from_minio(folderpath: str, local_dir: str) -> int:
    """
    MinIO folderpath 아래 이미지(jpg/png/jpeg)를 local_dir로 다운로드.
    하위 디렉토리 구조(cam_*/images/ 등)를 그대로 유지해 로컬에 받습니다.
    반환: 다운로드한 파일 수.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[annotation_job] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        return 0

    try:
        from minio import Minio
    except ImportError:
        print("[annotation_job] minio 라이브러리 없음: pip install minio", file=sys.stderr)
        return 0

    bucket, prefix = _parse_minio_folderpath(folderpath)
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    image_exts = (".jpg", ".jpeg", ".png")
    count = 0

    try:
        objects = client.list_objects(bucket, prefix=prefix + "/", recursive=True)
        for obj in objects:
            name_lower = obj.object_name.lower()
            if not any(name_lower.endswith(ext) for ext in image_exts):
                continue
            rel_path = obj.object_name[len(prefix):].lstrip("/")
            dest = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            client.fget_object(bucket, obj.object_name, dest)
            count += 1
    except Exception as e:
        print(f"[annotation_job] MinIO 다운로드 오류: {e}", file=sys.stderr)

    print(f"[annotation_job] MinIO 다운로드 완료: {count}개 이미지 → {local_dir}")
    return count


def run_annotation_job(
    division_idx: str,
    product_info: dict,
    is_cold: bool = False,
    *,
    gpu_id: Optional[int] = None,
    parallel_cams: bool = False,
) -> bool:
    """Download product images and run the Grounding DINO + SAM3 pipeline."""
    base = os.environ.get("ANNOTATION_BASE_DIR", str(_DEFAULT_ANNOTATION_BASE_DIR)).strip()
    if not base:
        print("[annotation_job] ANNOTATION_BASE_DIR missing", file=sys.stderr)
        return False

    foldername = str(product_info.get("foldername") or "").strip()
    folderpath = str(product_info.get("folderpath") or "").strip()
    product_idx = str(product_info.get("product_idx") or "").strip()
    train_product_idx = str(product_info.get("train_product_idx") or "").strip()
    brunch_name = str(product_info.get("brunch_name") or "").strip()
    product_name = str(
        product_info.get("product_eng_name")
        or product_info.get("productEngName")
        or _strip_foldername_timestamp(foldername)
    ).strip()

    if not foldername:
        print("[annotation_job] foldername missing", file=sys.stderr)
        return False

    run_name = _sam3_dino_run_name(division_idx)
    local_job_base = Path(base) / "_requests" / run_name if folderpath else Path(base)
    local_image_root = local_job_base / foldername
    if folderpath:
        downloaded = download_images_from_minio(folderpath, str(local_image_root))
        if downloaded == 0:
            print(f"[annotation_job] no images downloaded from MinIO: {folderpath}", file=sys.stderr)
            return False
    elif not local_image_root.is_dir():
        print(f"[annotation_job] local images and MinIO folderpath missing: {local_image_root}", file=sys.stderr)
        return False

    cams = sorted(
        p.name
        for p in local_image_root.iterdir()
        if p.is_dir() and re.fullmatch(r"cam_\d+", p.name) and _camera_has_images(p)
    )
    if not cams:
        print(f"[annotation_job] no camera images found: {local_image_root}", file=sys.stderr)
        return False

    runner = Path(os.environ.get("SAM3_DINO_RUNNER", str(_DEFAULT_SAM3_DINO_RUNNER)).strip())
    if not runner.is_file():
        print(f"[annotation_job] SAM3+DINO runner missing: {runner}", file=sys.stderr)
        return False

    runs_root = Path(
        os.environ.get("SAM3_DINO_RUNS_DIR", str(_DEFAULT_SAM3_DINO_RUNS_DIR)).strip()
        or str(_DEFAULT_SAM3_DINO_RUNS_DIR)
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    product_run_root = runs_root / run_name / foldername

    configured_gpus = [v.strip() for v in os.environ.get("PIPELINE_ANNOTATION_GPUS", "0").split(",") if v.strip()]
    effective_gpu = str(gpu_id) if gpu_id is not None else (configured_gpus[0] if configured_gpus else "")
    root_folder = "_".join(v for v in (brunch_name, train_product_idx, product_idx) if v)

    cmd = [
        sys.executable,
        str(runner),
        "--python-bin", sys.executable,
        "--img-root", str(local_job_base),
        "--runs-root", str(runs_root),
        "--product", foldername,
        "--cams", *cams,
        "--run-name", run_name,
        "--gpu", effective_gpu,
        "--backend", os.environ.get("SAM3_DINO_BACKEND", "video"),
        "--dino-model", os.environ.get("SAM3_DINO_MODEL", "IDEA-Research/grounding-dino-base"),
        "--hand-box-threshold", os.environ.get("SAM3_DINO_HAND_BOX_THRESHOLD", "0.25"),
        "--object-box-threshold", os.environ.get("SAM3_DINO_OBJECT_BOX_THRESHOLD", "0.22"),
        "--text-threshold", os.environ.get("SAM3_DINO_TEXT_THRESHOLD", "0.20"),
        "--sam3-confidence", os.environ.get("SAM3_DINO_CONFIDENCE", "0.35"),
        "--sam3-area-weight", os.environ.get("SAM3_DINO_AREA_WEIGHT", "0.15"),
        "--sam3-min-prompt-iou", os.environ.get("SAM3_DINO_MIN_PROMPT_IOU", "0.15"),
        "--sam3-min-area-ratio", os.environ.get("SAM3_DINO_MIN_AREA_RATIO", "0.15"),
        "--sam3-max-area-ratio", os.environ.get("SAM3_DINO_MAX_AREA_RATIO", "4.0"),
        "--temporal-object-iou-weight", os.environ.get("SAM3_DINO_TEMPORAL_OBJECT_IOU_WEIGHT", "0.20"),
        "--temporal-decay", os.environ.get("SAM3_DINO_TEMPORAL_DECAY", "0.75"),
        "--temporal-max-gap", os.environ.get("SAM3_DINO_TEMPORAL_MAX_GAP", "5"),
        "--min-temporal-compatibility", os.environ.get("SAM3_DINO_MIN_TEMPORAL_COMPATIBILITY", "0.30"),
        "--min-hand-object-relation", os.environ.get("SAM3_DINO_MIN_HAND_OBJECT_RELATION", "0.08"),
        "--object-solo-threshold", os.environ.get("SAM3_DINO_OBJECT_SOLO_THRESHOLD", "0.38"),
        "--hand-solo-threshold", os.environ.get("SAM3_DINO_HAND_SOLO_THRESHOLD", "0.40"),
        "--video-product-seed-threshold", os.environ.get("SAM3_DINO_VIDEO_PRODUCT_SEED_THRESHOLD", "0.24"),
        "--video-hand-seed-threshold", os.environ.get("SAM3_DINO_VIDEO_HAND_SEED_THRESHOLD", "0.30"),
        "--video-min-output-score", os.environ.get("SAM3_DINO_VIDEO_MIN_OUTPUT_SCORE", "0.40"),
        "--video-seed-support-radius", os.environ.get("SAM3_DINO_VIDEO_SEED_SUPPORT_RADIUS", "3"),
        "--cross-class-duplicate-iou", os.environ.get("SAM3_DINO_CROSS_CLASS_DUPLICATE_IOU", "0.88"),
        "--cross-class-duplicate-area-similarity", os.environ.get("SAM3_DINO_CROSS_CLASS_DUPLICATE_AREA_SIMILARITY", "0.85"),
        "--cross-class-duplicate-mask-iou", os.environ.get("SAM3_DINO_CROSS_CLASS_DUPLICATE_MASK_IOU", "0.80"),
        "--cross-class-hand-margin", os.environ.get("SAM3_DINO_CROSS_CLASS_HAND_MARGIN", "0.10"),
        "--background-pixel-threshold", os.environ.get("SAM3_DINO_BACKGROUND_PIXEL_THRESHOLD", "22"),
        "--background-min-hand-foreground", os.environ.get("SAM3_DINO_BACKGROUND_MIN_HAND_FOREGROUND", "0.06"),
        "--background-min-object-foreground", os.environ.get("SAM3_DINO_BACKGROUND_MIN_OBJECT_FOREGROUND", "0.08"),
        "--compact-hand-max-product-gap", os.environ.get("SAM3_DINO_COMPACT_HAND_MAX_PRODUCT_GAP", "0.18"),
        "--compact-hand-cluster-radius", os.environ.get("SAM3_DINO_COMPACT_HAND_CLUSTER_RADIUS", "0.22"),
        "--compact-hand-score-window", os.environ.get("SAM3_DINO_COMPACT_HAND_SCORE_WINDOW", "0.14"),
        "--compact-hand-min-margin", os.environ.get("SAM3_DINO_COMPACT_HAND_MIN_MARGIN", "0.08"),
        "--compact-hand-expansion", os.environ.get("SAM3_DINO_COMPACT_HAND_EXPANSION", "0.30"),
        "--hand-mask-support-expansion", os.environ.get("SAM3_DINO_HAND_MASK_SUPPORT_EXPANSION", "0.15"),
        "--hand-partial-mask-support-expansion", os.environ.get("SAM3_DINO_HAND_PARTIAL_MASK_SUPPORT_EXPANSION", "0.45"),
        "--hand-max-mask-support-expansion", os.environ.get("SAM3_DINO_HAND_MAX_MASK_SUPPORT_EXPANSION", "0.60"),
        "--hand-component-min-area-ratio", os.environ.get("SAM3_DINO_HAND_COMPONENT_MIN_AREA_RATIO", "0.04"),
        "--hand-component-max-gap", os.environ.get("SAM3_DINO_HAND_COMPONENT_MAX_GAP", "0.08"),
        "--hand-component-product-gap", os.environ.get("SAM3_DINO_HAND_COMPONENT_PRODUCT_GAP", "0.05"),
        "--hand-collapse-area-ratio", os.environ.get("SAM3_DINO_HAND_COLLAPSE_AREA_RATIO", "0.45"),
        "--hand-collapse-dimension-ratio", os.environ.get("SAM3_DINO_HAND_COLLAPSE_DIMENSION_RATIO", "0.50"),
        "--hand-collapse-history-min", os.environ.get("SAM3_DINO_HAND_COLLAPSE_HISTORY_MIN", "4"),
        "--hand-collapse-history-window", os.environ.get("SAM3_DINO_HAND_COLLAPSE_HISTORY_WINDOW", "9"),
        "--hand-recovery-min-area-ratio", os.environ.get("SAM3_DINO_HAND_RECOVERY_MIN_AREA_RATIO", "0.65"),
        "--hand-recovery-search-radius", os.environ.get("SAM3_DINO_HAND_RECOVERY_SEARCH_RADIUS", "60"),
        "--hand-recovery-context-frames", os.environ.get("SAM3_DINO_HAND_RECOVERY_CONTEXT_FRAMES", "45"),
        "--hand-recovery-max-anchors", os.environ.get("SAM3_DINO_HAND_RECOVERY_MAX_ANCHORS", "0"),
        "--hand-recovery-merge-gap", os.environ.get("SAM3_DINO_HAND_RECOVERY_MERGE_GAP", "5"),
        "--segmentation-fallback", os.environ.get("SAM3_DINO_SEGMENTATION_FALLBACK", "skip"),
        "--sam3-hand-prompt",
        os.environ.get(
            "SAM3_DINO_SAM3_HAND_PROMPT",
            "complete human hand from wrist to fingertips",
        ),
        "--product-name", product_name,
        "--product-idx", product_idx,
    ]
    if root_folder:
        cmd.extend(["--root-folder", root_folder])
    background_root = Path(base) / "_backgrounds" / division_idx
    configured_background_root = os.environ.get("SAM3_DINO_BACKGROUND_REFERENCE_ROOT", "").strip()
    if configured_background_root:
        background_root = Path(
            configured_background_root.format(
                division_idx=division_idx,
                brunch_name=brunch_name,
            )
        )
    if background_root.is_dir():
        cmd.extend(["--background-reference-root", str(background_root)])
    try:
        int(train_product_idx)
    except (TypeError, ValueError):
        pass
    else:
        cmd.extend(["--product-class-id", train_product_idx])

    hand_prompts = os.environ.get("SAM3_DINO_HAND_PROMPTS", "").strip()
    if hand_prompts:
        cmd.extend(["--hand-prompts", hand_prompts])
    product_prompts = os.environ.get("SAM3_DINO_PRODUCT_PROMPTS", "").strip()
    if product_prompts:
        cmd.extend(["--product-prompts", product_prompts])

    env = os.environ.copy()
    env["CLASS_NAME"] = foldername
    env["IS_COLD"] = "TRUE" if is_cold else "FALSE"
    env["PRODUCT_IDX"] = product_idx
    env["TRAIN_PRODUCT_IDX"] = train_product_idx
    env["BRUNCH_NAME"] = brunch_name
    if effective_gpu:
        env["CUDA_VISIBLE_DEVICES"] = effective_gpu
    timeout_sec = int(os.environ.get("ANNOTATION_JOB_TIMEOUT", "0")) or None

    try:
        ctx = get_request_context()
        job_log_dir = ctx.job_log_dir("annotation") if ctx else None
        print(
            f"[annotation_job] SAM3+DINO start: folder={foldername}, cams={cams}, "
            f"gpu={effective_gpu or 'cpu'}, run={run_name}, parallel_cams={parallel_cams}"
        )
        result = run_shell_job(
            cmd,
            job_type="annotation",
            log_name=foldername,
            cwd=str(runner.parent),
            env=env,
            timeout=timeout_sec,
            log_prefix="[annotation_job]",
            job_log_dir=job_log_dir,
            tee_stdout=False,
        )
        print(
            f"[annotation_job] SAM3+DINO logs: {product_run_root}/cam_*/run.log "
            f"and {job_log_dir or 'request jobs/annotation'}"
        )
        if result.returncode != 0:
            print(f"[annotation_job] SAM3+DINO exit code: {result.returncode}", file=sys.stderr)
            return False

        annotations_path = _find_sam3_dino_annotations(product_run_root)
        if annotations_path is None:
            return False
        if os.environ.get("ANNOTATION_SKIP_MINIO_UPLOAD", "false").strip().lower() == "true":
            print(f"[annotation_job] MinIO upload skipped: {annotations_path}")
            return True
        if not upload_review_package(product_info, annotations_path, local_image_root, cams):
            print(f"[annotation_job] web review package upload failed: {annotations_path}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[annotation_job] timeout: {foldername}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[annotation_job] exception: {exc}", file=sys.stderr)
        return False

def update_pnt_training_status(
    division_idx: str,
    product_idx: str,
    status: str,
) -> None:
    """
    (선택) PNT에 IF_07로 training_status 업데이트 POST.
    실제 연동 시 이 함수 구현 후 호출하면 됩니다.
    """
    print(
        f"[annotation_job] PNT IF_07 stub: division_idx={division_idx}, "
        f"product_idx={product_idx}, status={status}"
    )
