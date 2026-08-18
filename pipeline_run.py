"""product/created 요청(run) 단위 파이프라인 orchestration — Mongo 아님, 로컬 JSON 영속."""
from __future__ import annotations

import copy
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_STATE_DIR = os.path.join(_APP_ROOT, "logs", "pipeline_state")

PHASE_PENDING_SEGMENT = "pending_segment"

# product phase — DeviceTypeList.training_status(Mongo)와 별개, AI 서버 orchestration 전용
PHASE_PENDING_ANNOTATION = "pending_annotation"
PHASE_AWAITING_REVIEW = "awaiting_review"
PHASE_SEGMENTING = "segmenting"
PHASE_READY = "ready_for_training"
PHASE_SKIP_EXISTING = "skipped_existing"
PHASE_FAILED = "failed"

RUN_IN_PROGRESS = "in_progress"
RUN_TRAINING = "training"
RUN_DEPLOYED = "deployed"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_SUPERSEDED = "superseded"  # 레거시: 누적 병합 도입 전 상태 파일에만 남아 있음
RUN_MERGED = "merged"

_SKIP_ANNOTATION_PHASES = frozenset({
    PHASE_AWAITING_REVIEW,
    PHASE_SEGMENTING,
    PHASE_PENDING_SEGMENT,
    PHASE_READY,
    PHASE_SKIP_EXISTING,
})

_store_lock = threading.Lock()
_runs: Dict[str, dict] = {}
_loaded = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _state_dir() -> str:
    path = os.getenv("PNT_PIPELINE_STATE_DIR", _DEFAULT_STATE_DIR).strip()
    return path or _DEFAULT_STATE_DIR


def _safe_run_filename(run_id: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", (run_id or "unknown").strip()) or "unknown"
    return f"{safe}.json"


def _run_path(run_id: str) -> str:
    return os.path.join(_state_dir(), _safe_run_filename(run_id))


def _dt_to_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _dt_to_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dt_to_json(v) for v in value]
    return value


def _dt_from_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: _dt_from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dt_from_json(v) for v in value]
    return value


def _persist_run_unlocked(run_id: str) -> None:
    doc = _runs.get(run_id)
    if not doc:
        return
    root = _state_dir()
    os.makedirs(root, exist_ok=True)
    path = _run_path(run_id)
    tmp = f"{path}.tmp"
    payload = _dt_to_json(copy.deepcopy(doc))
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def _load_from_disk_unlocked() -> None:
    root = _state_dir()
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        if not name.endswith(".json") or name.endswith(".tmp"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as fp:
                doc = _dt_from_json(json.load(fp))
            run_id = str(doc.get("run_id") or "").strip()
            if run_id:
                _runs[run_id] = doc
        except (OSError, json.JSONDecodeError, TypeError):
            continue


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    with _store_lock:
        if _loaded:
            return
        _load_from_disk_unlocked()
        _loaded = True


def _division_key(division_idx: str, is_cold: bool) -> str:
    return f"{division_idx}:{'C' if is_cold else 'F'}"


def _product_entry(product_info: dict, phase: str) -> dict:
    return {
        "product_object_id": str(product_info.get("product_object_id") or ""),
        "product_idx": str(product_info.get("product_idx") or ""),
        "foldername": str(product_info.get("foldername") or ""),
        "train_product_idx": product_info.get("train_product_idx"),
        "is_new": product_info.get("is_new"),
        "phase": phase,
        "updated_at": _utcnow(),
    }


def _copy_run(doc: Optional[dict]) -> Optional[dict]:
    if doc is None:
        return None
    return copy.deepcopy(doc)


def has_new_annotation_artifact(product_info: dict) -> bool:
    endpoint = os.getenv("MINIO_ENDPOINT", "").strip()
    access_key = os.getenv("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.getenv("MINIO_SECRET_KEY", "").strip()
    if not (endpoint and access_key and secret_key):
        return False

    try:
        from minio import Minio
    except ImportError:
        return False

    bucket = os.getenv("MINIO_BUCKET", "chaiimage").strip() or "chaiimage"
    base_prefix = os.getenv("MINIO_BASE_PREFIX", "NewAnnotation").strip().strip("/") or "NewAnnotation"
    brunch_name = str(product_info.get("brunch_name") or "").strip()
    train_product_idx = str(product_info.get("train_product_idx") or "").strip()
    product_idx = str(product_info.get("product_idx") or "").strip()
    if not (brunch_name and train_product_idx and product_idx):
        return False

    secure = os.getenv("MINIO_SECURE", "false").strip().lower() == "true"
    root_folder = f"{brunch_name}_{train_product_idx}_{product_idx}"
    prefix = f"{base_prefix}/{root_folder}/"
    try:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        return any(client.list_objects(bucket, prefix=prefix, recursive=True))
    except Exception:
        return False


def merge_or_create_run_from_created(
    get_mongo_collection: Callable,
    *,
    run_id: str,
    if_id: str,
    division_idx: str,
    is_cold: bool,
    brunch_name: str,
    log_dir: str,
    pending_annotation: List[dict],
    pending_skip: List[dict],
    pending_segment: List[dict],
) -> tuple[Optional[dict], str, bool]:
    """Create a division run, or merge a repeated request into its active run.

    Repeated product/created events usually mean that products were added to the
    same division. Existing product phases must therefore never regress just
    because a newer request contains the full product list again.
    """
    del get_mongo_collection
    _ensure_loaded()
    if not run_id:
        return None, "", False

    requested: List[tuple[dict, str]] = []
    requested.extend((product, PHASE_PENDING_ANNOTATION) for product in pending_annotation)
    requested.extend((product, PHASE_SKIP_EXISTING) for product in pending_skip)
    requested.extend((product, PHASE_PENDING_SEGMENT) for product in pending_segment)

    missing_review_artifacts: set[str] = set()
    active_snapshot = find_active_run(None, division_idx, is_cold)
    if active_snapshot:
        for product in pending_annotation:
            product_object_id = str(product.get("product_object_id") or "")
            if (
                get_product_phase(active_snapshot, product_object_id) == PHASE_AWAITING_REVIEW
                and not has_new_annotation_artifact(product)
            ):
                missing_review_artifacts.add(product_object_id)

    now = _utcnow()
    key = _division_key(division_idx, is_cold)
    source_run_id = run_id
    with _store_lock:
        existing_run = _runs.get(run_id)
        if existing_run and existing_run.get("division_key") != key:
            base_run_id = f"{run_id}__{_safe_run_filename(key)[:-5]}"
            candidate = base_run_id
            suffix = 2
            while candidate in _runs and _runs[candidate].get("division_key") != key:
                candidate = f"{base_run_id}_{suffix}"
                suffix += 1
            run_id = candidate
        active_candidates = [
            doc
            for doc in _runs.values()
            if doc.get("division_key") == key and doc.get("run_status") == RUN_IN_PROGRESS
        ]
        active_candidates.sort(key=lambda doc: doc.get("created_at") or now, reverse=True)
        active = active_candidates[0] if active_candidates else None

        if active is None:
            products: List[dict] = []
            seen_product_ids: set[str] = set()
            for product, phase in requested:
                product_object_id = str(product.get("product_object_id") or "")
                if product_object_id and product_object_id in seen_product_ids:
                    continue
                products.append(_product_entry(product, phase))
                if product_object_id:
                    seen_product_ids.add(product_object_id)
            doc = {
                "run_id": run_id,
                "source_run_id": source_run_id,
                "if_id": if_id,
                "division_idx": division_idx,
                "division_key": key,
                "is_cold": is_cold,
                "brunch_name": brunch_name,
                "log_dir": log_dir,
                "run_status": RUN_IN_PROGRESS,
                "products": products,
                "created_at": now,
                "updated_at": now,
            }
            _runs[run_id] = doc
            _persist_run_unlocked(run_id)
            return _copy_run(doc), run_id, False

        effective_run_id = str(active.get("run_id") or "")
        by_product_id = {
            str(item.get("product_object_id") or ""): item
            for item in active.get("products") or []
            if str(item.get("product_object_id") or "")
        }

        for product, requested_phase in requested:
            product_object_id = str(product.get("product_object_id") or "")
            existing = by_product_id.get(product_object_id)
            if existing is None:
                entry = _product_entry(product, requested_phase)
                active.setdefault("products", []).append(entry)
                if product_object_id:
                    by_product_id[product_object_id] = entry
                continue

            refreshed = _product_entry(product, requested_phase)
            for field in ("product_idx", "foldername", "train_product_idx", "is_new"):
                value = refreshed.get(field)
                if value not in (None, ""):
                    existing[field] = value

            current_phase = str(existing.get("phase") or "")
            next_phase = current_phase
            if (
                current_phase == PHASE_AWAITING_REVIEW
                and requested_phase == PHASE_PENDING_ANNOTATION
                and product_object_id in missing_review_artifacts
            ):
                next_phase = PHASE_PENDING_ANNOTATION
            elif current_phase == PHASE_FAILED and requested_phase == PHASE_PENDING_ANNOTATION:
                next_phase = PHASE_PENDING_ANNOTATION
            elif requested_phase == PHASE_PENDING_SEGMENT and current_phase in {
                PHASE_PENDING_ANNOTATION,
                PHASE_FAILED,
                PHASE_AWAITING_REVIEW,
            }:
                next_phase = PHASE_PENDING_SEGMENT
            elif requested_phase == PHASE_SKIP_EXISTING and current_phase in {
                PHASE_PENDING_ANNOTATION,
                PHASE_FAILED,
            }:
                next_phase = PHASE_SKIP_EXISTING

            if next_phase != current_phase:
                existing["phase"] = next_phase
            existing["updated_at"] = now

        merged_request = {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "if_id": if_id,
            "division_idx": division_idx,
            "division_key": key,
            "is_cold": is_cold,
            "brunch_name": brunch_name,
            "log_dir": log_dir,
            "run_status": RUN_MERGED,
            "merged_into": effective_run_id,
            "products": [],
            "created_at": now,
            "updated_at": now,
        }

        active["updated_at"] = now
        if run_id != effective_run_id:
            _runs[run_id] = merged_request
            _persist_run_unlocked(run_id)
        _persist_run_unlocked(effective_run_id)
        return _copy_run(active), effective_run_id, run_id != effective_run_id


def find_active_run(
    get_mongo_collection: Callable,
    division_idx: str,
    is_cold: bool,
) -> Optional[dict]:
    del get_mongo_collection
    _ensure_loaded()
    key = _division_key(division_idx, is_cold)
    with _store_lock:
        candidates = [
            doc for doc in _runs.values()
            if doc.get("division_key") == key and doc.get("run_status") == RUN_IN_PROGRESS
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda d: d.get("created_at") or _utcnow(), reverse=True)
        return _copy_run(candidates[0])


def find_run_by_id(
    get_mongo_collection: Callable,
    run_id: str,
    division_idx: str = "",
) -> Optional[dict]:
    del get_mongo_collection
    _ensure_loaded()
    if not run_id:
        return None
    with _store_lock:
        doc = _runs.get(run_id)
        if doc is None:
            return None
        if division_idx and str(doc.get("division_idx")) != str(division_idx):
            return None
        return _copy_run(doc)


def list_recent_runs(
    division_idx: str,
    is_cold: bool,
    *,
    limit: int = 3,
) -> List[dict]:
    _ensure_loaded()
    key = _division_key(division_idx, is_cold)
    with _store_lock:
        candidates = [doc for doc in _runs.values() if doc.get("division_key") == key]
        candidates.sort(key=lambda d: d.get("created_at") or _utcnow(), reverse=True)
        return [_copy_run(d) for d in candidates[:limit]]


def _set_product_phase(run_id: str, product_object_id: str, phase: str) -> bool:
    _ensure_loaded()
    now = _utcnow()
    with _store_lock:
        doc = _runs.get(run_id)
        if not doc:
            return False
        matched = False
        for item in doc.get("products") or []:
            if str(item.get("product_object_id")) == str(product_object_id):
                item["phase"] = phase
                item["updated_at"] = now
                matched = True
                break
        if matched:
            doc["updated_at"] = now
            _persist_run_unlocked(run_id)
        return matched


def update_product_phase(
    get_mongo_collection: Callable,
    run_id: str,
    product_object_id: str,
    phase: str,
) -> bool:
    del get_mongo_collection
    if not run_id or not product_object_id:
        return False
    return _set_product_phase(run_id, product_object_id, phase)


def update_run_status(get_mongo_collection: Callable, run_id: str, run_status: str) -> bool:
    del get_mongo_collection
    _ensure_loaded()
    if not run_id:
        return False
    with _store_lock:
        doc = _runs.get(run_id)
        if not doc:
            return False
        doc["run_status"] = run_status
        doc["updated_at"] = _utcnow()
        _persist_run_unlocked(run_id)
        return True


def get_product_phase(run: Optional[dict], product_object_id: str) -> Optional[str]:
    if not run or not product_object_id:
        return None
    for item in run.get("products") or []:
        if str(item.get("product_object_id")) == str(product_object_id):
            return str(item.get("phase") or "")
    return None


def should_skip_annotation(
    get_mongo_collection: Callable,
    division_idx: str,
    is_cold: bool,
    product_object_id: str,
    run_id: str = "",
) -> bool:
    del get_mongo_collection
    run = find_active_run(None, division_idx, is_cold)
    if run_id and run and str(run.get("run_id")) != str(run_id):
        return True
    phase = get_product_phase(run, product_object_id)
    if phase is None:
        return False
    return phase not in (PHASE_PENDING_ANNOTATION, PHASE_FAILED)


def reset_failed_products_to_pending(
    get_mongo_collection: Callable,
    run_id: str,
) -> int:
    del get_mongo_collection
    _ensure_loaded()
    if not run_id:
        return 0
    with _store_lock:
        doc = _runs.get(run_id)
        if not doc:
            return 0
        reset = 0
        now = _utcnow()
        for item in doc.get("products") or []:
            if str(item.get("phase")) != PHASE_FAILED:
                continue
            item["phase"] = PHASE_PENDING_ANNOTATION
            item["updated_at"] = now
            reset += 1
        if reset:
            doc["updated_at"] = now
            _persist_run_unlocked(run_id)
        return reset


def run_summary(run: Optional[dict]) -> str:
    if not run:
        return "run=none"
    parts = []
    for item in run.get("products") or []:
        parts.append(f"{item.get('foldername')}:{item.get('phase')}")
    return f"run_id={run.get('run_id')} status={run.get('run_status')} [{', '.join(parts)}]"


def all_run_products_ready(run: Optional[dict]) -> bool:
    if not run:
        return True
    products = run.get("products") or []
    if not products:
        return True
    return all(
        str(p.get("phase")) in (PHASE_READY, PHASE_SKIP_EXISTING)
        for p in products
    )
