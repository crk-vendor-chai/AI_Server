"""
YOLO 학습·검증 job — `main.py` `_maybe_advance_pipeline` (division 전원 status=4) 시 호출.

흐름:
  1. `synthesis_job` (실패해도 학습 시도)
  2. MinIO `divisionSynthesis/` → 로컬 YOLO dataset + data.yaml
  3. Ultralytics fine-tune → augmented val → mAP50 vs `YOLO_VAL_THRESHOLD`

FP 억제 penalty loss 패치는 현재 운영 파이프라인에 적용하지 않는다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import random
import shutil
import sys
import tempfile

import torch
import ultralytics.nn.tasks as tasks
import ultralytics.utils.loss as loss_mod
import yaml

try:
    from .class_schema import (
        SYNTHETIC_CLASS_ID_OFFSET,
        canonical_class_key,
        canonical_class_name,
    )
except ImportError:
    from class_schema import (
        SYNTHETIC_CLASS_ID_OFFSET,
        canonical_class_key,
        canonical_class_name,
    )

PENALTY_WEIGHTS: dict[int, float] = {0: 2.0, 3: 2.0}

class NegativeClassPenaltyLoss(loss_mod.v8DetectionLoss):
    """Negative 앵커에서만 cls BCE에 클래스별 가중치 (FP 억제, focal loss 아님)."""

    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = loss_mod.make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))

        neg_weight = torch.ones(self.nc, device=self.device, dtype=dtype)
        for cls_id, w in PENALTY_WEIGHTS.items():
            if cls_id < self.nc:
                neg_weight[cls_id] = w
        cls_weight = torch.where(
            target_scores <= 0,
            neg_weight.view(1, 1, -1),
            torch.ones_like(neg_weight).view(1, 1, -1),
        )
        loss[1] = (cls_loss * cls_weight).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


def install_penalty_loss() -> None:
    """YOLO.train() 호출 전에 한 번 실행 — v8DetectionLoss를 penalty 버전으로 교체."""
    tasks.v8DetectionLoss = NegativeClassPenaltyLoss
    loss_mod.v8DetectionLoss = NegativeClassPenaltyLoss


def _parse_penalty_weights(s: str) -> dict[int, float]:
    """'0:3.0,3:2.0' → {0: 3.0, 3: 2.0}. weight 생략 시 2.0."""
    result = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            cls_str, w_str = part.split(":", 1)
            result[int(cls_str)] = float(w_str)
        else:
            result[int(part)] = 2.0
    return result

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_YOLO_MODEL_PATH = _PROJECT_ROOT / "yolov8n.pt"
_DEFAULT_YOLO_RUNS_DIR = _PROJECT_ROOT / "runs" / "detect" / "train"


def _run_segment_composition(division_idx: str, product_list: list, is_cold: bool = False) -> bool:
    """
    세그먼트 합성 실행.
    synthesis_job의 run_synthesis_job 호출.
    """
    try:
        from .synthesis_job import run_synthesis_job
    except ImportError:
        from synthesis_job import run_synthesis_job
    
    return run_synthesis_job(division_idx, product_list, is_cold=is_cold)


def _create_minio_client():
    from minio import Minio

    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[training] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        raise RuntimeError("MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _download_prefix_files(client, bucket: str, prefix: str, exts: tuple[str, ...], dest_root: Path) -> int:
    prefix = prefix.rstrip("/") + "/"
    object_names = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        name_lower = obj.object_name.lower()
        if any(name_lower.endswith(ext) for ext in exts):
            object_names.append(obj.object_name)
    if not object_names:
        return 0

    workers = min(int(os.getenv("YOLO_DOWNLOAD_WORKERS", "16")), max(1, len(object_names)))

    def _download_one(obj_name: str) -> None:
        rel = obj_name[len(prefix):].lstrip("/")
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj_name, str(dest))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_download_one, object_names))
    return len(object_names)


def _extract_class_id(train_product_idx, foldername: str) -> Optional[int]:
    try:
        if train_product_idx is not None:
            return int(train_product_idx)
    except Exception:
        pass
    if foldername:
        prefix = foldername.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
    return None


def _product_eng_name(product: dict) -> str:
    eng_name = (product.get("product_eng_name") or "").strip()
    if eng_name:
        return eng_name
    foldername = (product.get("foldername") or "").strip()
    if foldername and "_" in foldername:
        return foldername.split("_", 1)[1].rsplit("_", 2)[0]
    return ""


def _read_minio_yaml(client, bucket: str, object_name: str) -> dict:
    response = client.get_object(bucket, object_name)
    try:
        payload = yaml.safe_load(response.read()) or {}
    finally:
        response.close()
        response.release_conn()
    if not isinstance(payload, dict):
        raise RuntimeError(f"검수 data.yaml 형식이 dict가 아닙니다: {object_name}")
    return payload


def _normalize_yaml_names(raw_names, object_name: str) -> dict[int, str]:
    if isinstance(raw_names, list):
        result = {i: str(name).strip() for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        result = {}
        for raw_id, raw_name in raw_names.items():
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"검수 data.yaml names 키가 정수가 아닙니다: {object_name}, key={raw_id!r}"
                ) from exc
            result[class_id] = str(raw_name).strip()
    else:
        raise RuntimeError(f"검수 data.yaml에 names가 없습니다: {object_name}")

    if not result or min(result) != 0 or sorted(result) != list(range(max(result) + 1)):
        raise RuntimeError(f"검수 data.yaml names가 0부터 연속적이지 않습니다: {object_name}")
    if any(not name for name in result.values()):
        raise RuntimeError(f"검수 data.yaml names에 빈 클래스명이 있습니다: {object_name}")
    return result


def _find_review_yaml_object(client, bucket: str, product: dict) -> Optional[str]:
    review_root = os.getenv("YOLO_REVIEW_PREFIX", "productAnnotation").strip().strip("/")
    foldername = (product.get("foldername") or "").strip()
    train_product_idx = str(product.get("train_product_idx") or "").strip()
    product_eng_name = _product_eng_name(product)

    def _yaml_objects(prefix: str, *, directory_prefix: bool) -> list[str]:
        if not prefix:
            return []
        search_prefix = prefix.rstrip("/") + "/" if directory_prefix else prefix
        return sorted(
            obj.object_name
            for obj in client.list_objects(bucket, prefix=search_prefix, recursive=True)
            if Path(obj.object_name).name.lower() in {"data.yaml", "data.yml"}
        )

    if foldername:
        exact = _yaml_objects(f"{review_root}/{foldername}", directory_prefix=True)
        if exact:
            return exact[-1]

    if train_product_idx and product_eng_name:
        wildcard = _yaml_objects(
            f"{review_root}/{train_product_idx}_{product_eng_name}_",
            directory_prefix=False,
        )
        if wildcard:
            return wildcard[-1]
    return None


def _review_dataset_key(product: dict) -> str:
    train_product_idx = _extract_class_id(
        product.get("train_product_idx"),
        product.get("foldername"),
    )
    product_eng_name = _product_eng_name(product)
    if train_product_idx is None or not product_eng_name:
        raise RuntimeError(f"상품 학습 식별자 또는 영문명이 없습니다: {product}")
    return f"syn_{train_product_idx}_{product_eng_name}"


def _load_review_class_schemas(
    client,
    bucket: str,
    product_list: list,
) -> tuple[dict[str, dict[int, str]], dict[str, str]]:
    """상품별 CVAT export 스키마를 독립적으로 읽는다.

    CVAT 라벨을 삭제하거나 다시 만들면 YOLO export의 숫자 class_id가
    재정렬될 수 있으므로 서로 다른 data.yaml을 숫자 ID로 병합하지 않는다.
    """
    schemas: dict[str, dict[int, str]] = {}
    sources: dict[str, str] = {}
    for product in product_list:
        dataset_key = _review_dataset_key(product)
        object_name = _find_review_yaml_object(client, bucket, product)
        if not object_name:
            raise RuntimeError(
                "productAnnotation에서 검수 data.yaml을 찾지 못했습니다: "
                f"{dataset_key}"
            )
        names = _normalize_yaml_names(
            _read_minio_yaml(client, bucket, object_name).get("names"),
            object_name,
        )
        if canonical_class_key(names[0]) != "hand":
            raise RuntimeError(
                f"검수 data.yaml의 class 0이 hand가 아닙니다: {names[0]!r} "
                f"({object_name})"
            )
        schemas[dataset_key] = names
        sources[dataset_key] = object_name

    if not schemas:
        raise RuntimeError("productAnnotation에서 검수 data.yaml을 찾지 못했습니다.")
    return schemas, sources


def _load_latest_deployed_class_names(
    client,
    bucket: str,
    division_idx: str,
    is_cold: bool,
) -> tuple[list[str], Optional[str]]:
    """최근 배포 모델의 names를 안정적인 전역 class_id 기준으로 사용한다."""
    prefix = f"trained_model/{division_idx}_{'C' if is_cold else 'F'}/"
    candidates = [
        obj
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True)
        if obj.object_name.lower().endswith(".pt")
    ]
    if not candidates:
        return [], None

    latest = max(
        candidates,
        key=lambda obj: (
            getattr(obj, "last_modified", None).isoformat()
            if getattr(obj, "last_modified", None)
            else "",
            obj.object_name,
        ),
    )
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="yolo_schema_") as tmp:
        model_path = Path(tmp) / "model.pt"
        client.fget_object(bucket, latest.object_name, str(model_path))
        raw_names = YOLO(str(model_path)).names
    names_dict = _normalize_yaml_names(raw_names, latest.object_name)
    names = [canonical_class_name(names_dict[i]) for i in range(len(names_dict))]
    if canonical_class_key(names[0]) != "hand":
        raise RuntimeError(
            f"배포 모델의 class 0이 hand가 아닙니다: {names[0]!r} ({latest.object_name})"
        )
    return names, latest.object_name


def _build_canonical_class_names(
    deployed_names: list[str],
    review_schemas: dict[str, dict[int, str]],
) -> list[str]:
    """배포 모델 순서를 보존하고 검수 export의 새 클래스만 뒤에 추가한다."""
    if deployed_names:
        names = [canonical_class_name(name) for name in deployed_names]
    else:
        # 최초 학습은 가장 많은 클래스를 가진 검수 export를 기준으로 시작한다.
        base = max(review_schemas.values(), key=lambda schema: len(schema))
        names = [canonical_class_name(base[i]) for i in range(len(base))]

    name_to_id: dict[str, int] = {}
    for class_id, name in enumerate(names):
        key = canonical_class_key(name)
        if key in name_to_id:
            raise RuntimeError(
                f"기준 클래스명 중복: {name!r}, class_ids={name_to_id[key]},{class_id}"
            )
        name_to_id[key] = class_id

    for schema in review_schemas.values():
        for source_id in sorted(schema):
            name = canonical_class_name(schema[source_id])
            key = canonical_class_key(name)
            if key not in name_to_id:
                name_to_id[key] = len(names)
                names.append(name)

    if not names or canonical_class_key(names[0]) != "hand":
        raise RuntimeError("기준 클래스 스키마의 class 0은 hand여야 합니다.")
    return names


def _build_source_class_maps(
    product_list: list,
    names: list[str],
    review_schemas: dict[str, dict[int, str]],
) -> tuple[dict[str, dict[int, int]], dict[int, int], dict[int, int]]:
    """각 CVAT export ID와 합성용 train_product_idx를 전역 class_id로 연결한다."""
    name_to_ids: dict[str, list[int]] = {}
    for class_id, name in enumerate(names):
        name_to_ids.setdefault(canonical_class_key(name), []).append(class_id)

    review_source_maps: dict[str, dict[int, int]] = {}
    for dataset_key, schema in review_schemas.items():
        source_map: dict[int, int] = {}
        for source_id, source_name in schema.items():
            matched_ids = name_to_ids.get(canonical_class_key(source_name), [])
            if len(matched_ids) != 1:
                raise RuntimeError(
                    "검수 클래스명을 기준 스키마에서 유일하게 찾지 못했습니다: "
                    f"dataset={dataset_key}, source_id={source_id}, "
                    f"name={source_name!r}, matched_class_ids={matched_ids}"
                )
            source_map[source_id] = matched_ids[0]
        review_source_maps[dataset_key] = source_map

    synthetic_source_map: dict[int, int] = {}
    train_product_idx_map: dict[int, int] = {}
    for product in product_list:
        train_product_idx = _extract_class_id(
            product.get("train_product_idx"),
            product.get("foldername"),
        )
        product_eng_name = _product_eng_name(product)
        if train_product_idx is None or not product_eng_name:
            raise RuntimeError(f"상품 학습 식별자 또는 영문명이 없습니다: {product}")

        matched_ids = name_to_ids.get(canonical_class_key(product_eng_name), [])
        if len(matched_ids) != 1:
            raise RuntimeError(
                "기준 클래스 스키마에서 상품명을 유일하게 찾지 못했습니다: "
                f"train_product_idx={train_product_idx}, name={product_eng_name!r}, "
                f"matched_class_ids={matched_ids}"
            )
        target_class_id = matched_ids[0]
        existing = synthetic_source_map.get(train_product_idx)
        if existing is not None and existing != target_class_id:
            raise RuntimeError(
                "train_product_idx가 서로 다른 상품 클래스를 가리킵니다: "
                f"train_product_idx={train_product_idx}, previous={existing}, "
                f"product_class_id={target_class_id}"
            )
        synthetic_source_map[train_product_idx] = target_class_id
        train_product_idx_map[target_class_id] = train_product_idx
    return review_source_maps, synthetic_source_map, train_product_idx_map


def _remap_label_class_ids(
    labels_root: Path,
    review_source_maps: dict[str, dict[int, int]],
    synthetic_source_map: dict[int, int],
) -> tuple[int, int, dict]:
    """작업별 CVAT ID와 합성 ID를 안정적인 전역 class_id로 변환한다."""
    checked = 0
    dropped = 0
    dist: dict[int, int] = {}
    for txt_path in labels_root.rglob("*.txt"):
        relative_parts = txt_path.relative_to(labels_root).parts
        dataset_key = relative_parts[0] if relative_parts else ""
        review_source_map = review_source_maps.get(dataset_key)
        if review_source_map is None:
            raise RuntimeError(f"라벨의 검수 클래스 스키마를 찾지 못했습니다: {txt_path}")
        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                pid = int(float(parts[0]))
            except ValueError:
                dropped += 1
                continue

            if pid >= SYNTHETIC_CLASS_ID_OFFSET:
                train_product_idx = pid - SYNTHETIC_CLASS_ID_OFFSET
                class_id = synthetic_source_map.get(train_product_idx)
            else:
                # 기존 CVAT 라벨은 이 export의 data.yaml로 해석한다.
                class_id = review_source_map.get(pid)
                if class_id is None:
                    # 과거 합성 산출물은 train_product_idx를 그대로 기록했다.
                    class_id = synthetic_source_map.get(pid)
            if class_id is None:
                dropped += 1
                continue
            dist[class_id] = dist.get(class_id, 0) + 1
            new_lines.append(" ".join([str(class_id)] + parts[1:]))
        # 라벨이 하나도 없으면 빈 파일로 남긴다 (negative sample)
        txt_path.write_text("\n".join(new_lines), encoding="utf-8")
        checked += 1
    return checked, dropped, dist


def _build_yolo_dataset_from_minio(
    division_idx: str,
    product_list: list,
    dataset_root: Path,
    is_cold: bool = False,
) -> tuple[Path, int, Path, Path, list[str]]:
    """
    MinIO에서 합성 데이터만 다운로드 및 train/val 분할.
    
    1. 합성 데이터: divisionSynthesis/{divisionidx}_{trainProductIdx}_{engname}/
    2. train/val 분할 (기본 0.95/0.05)
    
    참고: 원본 데이터는 합성 시에만 사용되고, 학습에는 합성 데이터만 사용됨.
    """
    client = _create_minio_client()
    bucket = os.getenv("YOLO_MINIO_BUCKET", "chaiimage").strip()
    synthesis_root = os.getenv("YOLO_SYNTHESIS_PREFIX", "divisionSynthesis").strip().strip("/")

    # 임시 병합 디렉토리 (train/val 분할 전)
    temp_img_root = dataset_root / "temp" / "images"
    temp_lbl_root = dataset_root / "temp" / "labels"
    temp_img_root.mkdir(parents=True, exist_ok=True)
    temp_lbl_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_labels = 0

    # 합성 데이터만 다운로드
    for product in product_list:
        train_product_idx = str(product.get("train_product_idx") or "").strip()
        product_eng_name = (product.get("product_eng_name") or "").strip()
        if not train_product_idx or not product_eng_name:
            continue

        # divisionSynthesis/{divisionidx}_{trainProductIdx}_{engname}/
        syn_prefix = f"{synthesis_root}/{division_idx}_{train_product_idx}_{product_eng_name}"
        syn_dest_img = temp_img_root / f"syn_{train_product_idx}_{product_eng_name}"
        syn_dest_lbl = temp_lbl_root / f"syn_{train_product_idx}_{product_eng_name}"

        syn_img_count = _download_prefix_files(
            client=client,
            bucket=bucket,
            prefix=syn_prefix,
            exts=(".jpg", ".jpeg", ".png"),
            dest_root=syn_dest_img,
        )
        syn_lbl_count = _download_prefix_files(
            client=client,
            bucket=bucket,
            prefix=syn_prefix,
            exts=(".txt",),
            dest_root=syn_dest_lbl,
        )
        total_images += syn_img_count
        total_labels += syn_lbl_count

    print(f"[training] 합성 데이터: images={total_images}, labels={total_labels}")

    if total_images == 0:
        raise RuntimeError("합성 데이터가 비어 있습니다. 먼저 segment 합성을 실행하세요.")

    review_schemas, yaml_sources = _load_review_class_schemas(client, bucket, product_list)
    deployed_names, deployed_source = _load_latest_deployed_class_names(
        client,
        bucket,
        division_idx,
        is_cold,
    )
    names = _build_canonical_class_names(deployed_names, review_schemas)
    review_source_maps, synthetic_source_map, train_product_idx_map = (
        _build_source_class_maps(product_list, names, review_schemas)
    )
    print(f"[training] 검수 data.yaml 로드: {list(yaml_sources.values())}")
    print(
        f"[training] 기준 클래스 스키마: "
        f"{deployed_source or 'review data.yaml (initial training)'}, nc={len(names)}"
    )
    print(
        f"[training] 클래스 매핑(train_product_idx -> 학습 class_id): "
        f"{dict(sorted((pid, class_id) for class_id, pid in train_product_idx_map.items()))}"
    )

    checked, dropped, dist = _remap_label_class_ids(
        temp_lbl_root,
        review_source_maps,
        synthetic_source_map,
    )
    print(
        f"[training] 라벨 변환: 파일={checked}, 제거된 박스={dropped}, "
        f"클래스 분포={dict(sorted(dist.items()))}"
    )
    if dropped:
        print(
            f"[training] 경고: 기준 스키마로 변환할 수 없는 박스 {dropped}개를 제거했습니다",
            file=sys.stderr,
        )

    # train/val 분할
    val_split_ratio = float(os.getenv("YOLO_VAL_SPLIT_RATIO", "0.05"))
    train_img_root = dataset_root / "train" / "images"
    train_lbl_root = dataset_root / "train" / "labels"
    val_img_root = dataset_root / "val" / "images"
    val_lbl_root = dataset_root / "val" / "labels"
    train_img_root.mkdir(parents=True, exist_ok=True)
    train_lbl_root.mkdir(parents=True, exist_ok=True)
    val_img_root.mkdir(parents=True, exist_ok=True)
    val_lbl_root.mkdir(parents=True, exist_ok=True)

    all_images = list(temp_img_root.rglob("*.jpg")) + list(temp_img_root.rglob("*.png"))
    random.shuffle(all_images)
    n_val = int(len(all_images) * val_split_ratio)

    for i, img_path in enumerate(all_images):
        is_val = i < n_val
        target_img_dir = val_img_root if is_val else train_img_root
        target_lbl_dir = val_lbl_root if is_val else train_lbl_root

        # 이미지 이동
        rel_path = img_path.relative_to(temp_img_root)
        dest_img = target_img_dir / rel_path
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_path), str(dest_img))

        # 라벨 이동
        lbl_path = temp_lbl_root / rel_path.with_suffix(".txt")
        if lbl_path.exists():
            dest_lbl = target_lbl_dir / rel_path.with_suffix(".txt")
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(lbl_path), str(dest_lbl))

    # 임시 디렉토리 삭제
    shutil.rmtree(dataset_root / "temp", ignore_errors=True)

    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(train_img_root),
                "val": str(val_img_root),
                "nc": len(names),
                "names": {i: name for i, name in enumerate(names)},
                "train_product_idx_map": train_product_idx_map,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    active_names = {class_id: names[class_id] for class_id in sorted(dist)}
    print(
        f"[training] dataset 준비 완료: total_images={total_images}, total_labels={total_labels}, "
        f"train={len(all_images) - n_val}, val={n_val}, nc={len(names)}, "
        f"class_names={active_names}"
    )
    return data_yaml, total_images, train_img_root, train_lbl_root, names


def _train_yolo(
    division_idx: str,
    is_cold: bool,
    data_yaml: Path,
    use_penalty: bool = False,
) -> Path:
    from ultralytics import YOLO

    model_path = os.getenv(
        "YOLO_MODEL_PATH",
        str(_DEFAULT_YOLO_MODEL_PATH),
    ).strip()
    epochs = int(os.getenv("YOLO_EPOCHS", "450"))
    imgsz = int(os.getenv("YOLO_IMGSZ", "480"))
    batch = int(os.getenv("YOLO_BATCH", "128"))
    workers = int(os.getenv("YOLO_WORKERS", "8"))
    patience = int(os.getenv("YOLO_PATIENCE", "30"))
    optimizer = os.getenv("YOLO_OPTIMIZER", "AdamW").strip()
    project = os.getenv(
        "YOLO_RUNS_DIR",
        str(_DEFAULT_YOLO_RUNS_DIR),
    ).strip()
    # 옵션 조합별로 결과 디렉터리가 구분되도록 접미사를 붙인다.
    # deploy_job._get_latest_best_pt() 는 "{division}_{C|F}_" 접두로 탐색하므로
    # 접미사가 붙어도 배포 탐색에는 영향이 없다.
    suffix = "_use_penalty" if use_penalty else ""
    run_name = (
        f"{division_idx}_{'C' if is_cold else 'F'}{suffix}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device = os.getenv("YOLO_DEVICE", "0").strip()
    resume = os.getenv("YOLO_RESUME", "false").strip().lower() == "true"
    hsv_h = float(os.getenv("YOLO_HSV_H", "0.05"))
    hsv_s = float(os.getenv("YOLO_HSV_S", "0.55"))
    hsv_v = float(os.getenv("YOLO_HSV_V", "0.6"))
    mosaic = float(os.getenv("YOLO_MOSAIC", "0.6"))
    mixup = float(os.getenv("YOLO_MIXUP", "0.3"))

    custom_transforms = None
    try:
        import albumentations as A

        custom_transforms = [
            A.ToGray(p=0.15),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=5,
                val_shift_limit=0,
                p=0.25,
            ),
        ]
    except Exception:
        pass

    model = YOLO(model_path)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "workers": workers,
        "optimizer": optimizer,
        "resume": resume,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "mosaic": mosaic,
        "mixup": mixup,
        "cache": True,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "plots": True,
    }
    if custom_transforms is not None:
        train_kwargs["augmentations"] = custom_transforms
    if device:
        train_kwargs["device"] = device

    model.train(**train_kwargs)
    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / run_name))
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        best_model = save_dir / "weights" / "last.pt"
    print(f"[training] YOLO 학습 완료: run={save_dir}, best={best_model}")
    return best_model


def _build_augmented_val_yaml(
    train_img_root: Path,
    train_lbl_root: Path,
    dataset_root: Path,
    names: list[str],
) -> Path:
    """val 전용 약한 augment 데이터셋 생성 후 yaml 반환."""
    val_img_root = dataset_root / "val_aug" / "images"
    val_lbl_root = dataset_root / "val_aug" / "labels"
    val_img_root.mkdir(parents=True, exist_ok=True)
    val_lbl_root.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in train_img_root.rglob("*") if p.suffix.lower() in image_exts]
    if not image_paths:
        raise RuntimeError("val augment 생성 실패: 원본 이미지가 없습니다.")

    max_images = int(os.getenv("YOLO_VAL_AUG_MAX_IMAGES", "0"))
    if max_images > 0 and len(image_paths) > max_images:
        image_paths = random.sample(image_paths, max_images)

    use_aug = True
    try:
        import albumentations as A
        import cv2

        aug = A.Compose([
            A.ToGray(p=0.05),
            A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.03, p=0.15),
            A.GaussianBlur(blur_limit=(3, 3), p=0.08),
        ])
    except Exception:
        use_aug = False
        aug = None
        cv2 = None

    for src_img in image_paths:
        rel = src_img.relative_to(train_img_root)
        dst_img = val_img_root / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        if use_aug:
            img = cv2.imread(str(src_img))
            if img is not None:
                out = aug(image=img)["image"]
                cv2.imwrite(str(dst_img), out)
            else:
                shutil.copy2(src_img, dst_img)
        else:
            shutil.copy2(src_img, dst_img)

        src_lbl = train_lbl_root / rel.with_suffix(".txt")
        if src_lbl.exists():
            dst_lbl = val_lbl_root / rel.with_suffix(".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_lbl, dst_lbl)

    val_yaml = dataset_root / "val_aug.yaml"
    val_yaml.write_text(
        f"train: {train_img_root}\n"
        f"val: {val_img_root}\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    print(f"[validation] val_aug 생성 완료: images={len(image_paths)}, path={val_img_root}")
    return val_yaml


def _validate_yolo(best_model: Path, data_yaml: Path) -> float:
    """학습된 모델을 val(전용 aug 데이터셋)로 평가하고 mAP50 반환."""
    from ultralytics import YOLO

    imgsz = int(os.getenv("YOLO_VAL_IMGSZ", os.getenv("YOLO_IMGSZ", "480")))
    batch = int(os.getenv("YOLO_VAL_BATCH", os.getenv("YOLO_BATCH", "128")))
    device = os.getenv("YOLO_DEVICE", "1").strip()
    project = os.getenv(
        "YOLO_VAL_RUNS_DIR",
        str(Path(__file__).resolve().parent.parent / "runs" / "detect" / "runs" / "val"),
    ).strip()
    train_run_name = best_model.parent.parent.name or "model"
    run_name = f"{train_run_name}_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model = YOLO(str(best_model))
    val_kwargs = {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "augment": False,
        "plots": False,
        "project": project,
        "name": run_name,
        "exist_ok": True,
    }
    if device:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)
    if hasattr(metrics, "box") and hasattr(metrics.box, "map50"):
        score = float(metrics.box.map50)
    else:
        result_dict = getattr(metrics, "results_dict", {}) or {}
        score = float(result_dict.get("metrics/mAP50(B)", 0.0))
    print(f"[validation] mAP50={score:.4f}")
    return score


def run_training_job(division_idx: str, product_list: list, is_cold: bool = False, use_penalty: bool = False) -> dict:
    """MinIO 기반 YOLO 학습 + 검증 job 실행."""
    print(f"[training] 학습 job 시작 - division_idx={division_idx}, product_count={len(product_list)}")
    if not product_list:
        print("[training] product_list 비어있음, 종료")
        return {"train_success": False, "validation_score": None, "validation_passed": False}

    # 합성 실행 (실패해도 학습은 진행)
    synthesis_ok = _run_segment_composition(division_idx, product_list, is_cold=is_cold)
    if not synthesis_ok:
        print("[training] 합성 실패했지만 학습은 진행합니다")
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"yolo_train_{division_idx}_"))
    try:
        if not use_penalty:
            data_yaml, _, train_img_root, train_lbl_root, names = _build_yolo_dataset_from_minio(
                division_idx,
                product_list,
                tmp_dir,
                is_cold=is_cold,
            )
            best_model = _train_yolo(division_idx, is_cold, data_yaml, use_penalty=False)
            val_aug_yaml = _build_augmented_val_yaml(train_img_root, train_lbl_root, tmp_dir, names)
            score = _validate_yolo(best_model, val_aug_yaml)
            threshold = float(os.getenv("YOLO_VAL_THRESHOLD", "0.8"))
            passed = score >= threshold
            print(f"[validation] threshold={threshold:.4f}, passed={passed}")
            return {"train_success": True, "validation_score": score, "validation_passed": passed}
        else:
            global PENALTY_WEIGHTS
            default_pw = ",".join(f"{c}:{w}" for c, w in PENALTY_WEIGHTS.items())
            data_yaml, _, train_img_root, train_lbl_root, names = _build_yolo_dataset_from_minio(
                division_idx,
                product_list,
                tmp_dir,
                is_cold=is_cold,
            )
            PENALTY_WEIGHTS = _parse_penalty_weights(default_pw)
            install_penalty_loss()
            best_model = _train_yolo(division_idx, is_cold, data_yaml, use_penalty)
            val_aug_yaml = _build_augmented_val_yaml(train_img_root, train_lbl_root, tmp_dir, names)
            score = _validate_yolo(best_model, val_aug_yaml)
            threshold = float(os.getenv("YOLO_VAL_THRESHOLD", "0.8"))
            passed = score >= threshold
            print(f"[validation] threshold={threshold:.4f}, passed={passed}")
            return {"train_success": True, "validation_score": score, "validation_passed": passed}
    except Exception as e:
        print(f"[training] 학습 실패: {e}")
        return {"train_success": False, "validation_score": None, "validation_passed": False}
    finally:
        if os.getenv("YOLO_KEEP_TMP", "false").strip().lower() == "true":
            print(f"[training] 임시 데이터 유지: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(run_training_job())