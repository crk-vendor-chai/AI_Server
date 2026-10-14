from fastapi import BackgroundTasks, FastAPI

# ===== [추가] validation/exception 핸들링을 위한 import =====
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Dict, TextIO
from datetime import datetime
import socket
import os
import sys
import json
import uuid
import threading
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

try:
    from .annotation_job import run_annotation_job
    from .annotation_batch_queue import annotation_batch_queue
    from .gpu_scheduler import GpuSlotTimeout as _GpuSlotTimeout, get_gpu_slot_pool
    from .http_logging import HttpTransactionLoggingMiddleware, log_event, request_payload_from_state
    from .training_job import run_training_job
    from .segment_job import run_segment_job
    from .deploy_job import run_deploy_job
    from .request_log import RequestLogContext, bind_request_context, init_request_log, pipeline_log, resume_request_log
    from .pipeline_run import (
        PHASE_AWAITING_REVIEW,
        PHASE_FAILED,
        PHASE_PENDING_ANNOTATION,
        PHASE_PENDING_SEGMENT,
        PHASE_READY,
        PHASE_SEGMENTING,
        RUN_COMPLETED,
        RUN_DEPLOYED,
        RUN_FAILED,
        RUN_IN_PROGRESS,
        RUN_TRAINING,
        find_active_run,
        has_new_annotation_artifact,
        merge_or_create_run_from_created,
        run_summary,
        reset_failed_products_to_pending,
        should_skip_annotation,
        update_product_phase,
        update_run_status,
    )
except ImportError:
    from annotation_job import run_annotation_job
    from annotation_batch_queue import annotation_batch_queue
    from gpu_scheduler import GpuSlotTimeout as _GpuSlotTimeout, get_gpu_slot_pool
    from http_logging import HttpTransactionLoggingMiddleware, log_event, request_payload_from_state
    from training_job import run_training_job
    from segment_job import run_segment_job
    from deploy_job import run_deploy_job
    from request_log import RequestLogContext, bind_request_context, init_request_log, pipeline_log, resume_request_log
    from pipeline_run import (
        PHASE_AWAITING_REVIEW,
        PHASE_FAILED,
        PHASE_PENDING_ANNOTATION,
        PHASE_PENDING_SEGMENT,
        PHASE_READY,
        PHASE_SEGMENTING,
        RUN_COMPLETED,
        RUN_DEPLOYED,
        RUN_FAILED,
        RUN_IN_PROGRESS,
        RUN_TRAINING,
        find_active_run,
        has_new_annotation_artifact,
        merge_or_create_run_from_created,
        run_summary,
        reset_failed_products_to_pending,
        should_skip_annotation,
        update_product_phase,
        update_run_status,
    )

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_APP_ROOT, ".env"))
_DEFAULT_LOG_DIR = os.path.join(_APP_ROOT, "logs")


class _ThreadSafeTee:
    """stdout/stderr를 터미널과 logs/api.log에 동시 기록."""

    def __init__(self, stream, log_file: TextIO):
        self._stream = stream
        self._log = log_file
        self._lock = threading.Lock()

    def write(self, data):
        if not data:
            return
        with self._lock:
            self._stream.write(data)
            self._stream.flush()
            self._log.write(data)
            self._log.flush()

    def flush(self):
        with self._lock:
            self._stream.flush()
            self._log.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


def _setup_app_logging() -> str:
    """
    API/job print 로그를 0_pnt/logs/ 아래 파일에도 저장.
    PNT_LOG_DIR, PNT_LOG_FILE, PNT_LOG_TO_FILE(false면 파일 미사용)로 제어.
    """
    enabled = os.getenv("PNT_LOG_TO_FILE", "true").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return ""

    log_dir = os.getenv("PNT_LOG_DIR", _DEFAULT_LOG_DIR).strip() or _DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, os.getenv("PNT_LOG_FILE", "api.log").strip() or "api.log")

    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _ThreadSafeTee(sys.stdout, log_fp)
    sys.stderr = _ThreadSafeTee(sys.stderr, log_fp)
    print(f"[log] file logging enabled: {log_path}")
    return log_path


_APP_LOG_PATH = _setup_app_logging()

app = FastAPI(title="CRK AI Event API", version="1.0.0")
app.add_middleware(HttpTransactionLoggingMiddleware)
_mongo_client: Optional[MongoClient] = None
_pipeline_locks: Dict[str, threading.Lock] = {}
_annotation_locks: Dict[str, threading.Lock] = {}
_annotation_active_run: Dict[str, str] = {}

# -----------------------------
# Common models (PNT-style)
# -----------------------------
class Header(BaseModel):
    IF_ID: str = Field(..., description="Interface ID")
    IF_SYSID: str = Field(..., description="Request unique id (trace id)")
    IF_HOST: str = Field(..., description="Caller host/system name")
    IF_DATE: str = Field(..., description="YYYYMMDDhhmmss")

class ResultData(BaseModel):
    result_cd: str = Field(..., description="S or F")
    result_msg: str = Field(..., description="message")

class ResponseEnvelope(BaseModel):
    HEADER: Header
    DATA: ResultData

def make_response(req_header: Header, result_cd: str, result_msg: str) -> ResponseEnvelope:
    # 응답 헤더는 요청 값을 최대한 유지하면서, HOST/DATE만 우리 서버 기준으로 바꿔줌
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    resp_header = Header(
        IF_ID=req_header.IF_ID,
        IF_SYSID=req_header.IF_SYSID,
        IF_HOST=socket.gethostname(),  # 우리 서버 호스트명
        IF_DATE=now,
    )
    return ResponseEnvelope(
        HEADER=resp_header,
        DATA=ResultData(result_cd=result_cd, result_msg=result_msg),
    )

def make_error_response(req_header: Header, status_code: int, result_msg: str) -> JSONResponse:
    response = make_response(req_header, "F", result_msg)
    return JSONResponse(status_code=status_code, content=response.model_dump())

# ===== [추가] FAIL 응답을 우리 포맷(HEADER/DATA + result_cd/result_msg)으로 통일 =====
def _now_yyyymmddhhmmss() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")

def _safe_req_header(body: dict) -> Header:
    """
    요청이 잘못되어도(필드 누락/타입 오류 등) 가능한 한 요청 HEADER를 응답에 담기 위해 안전하게 추출.
    요청 JSON 자체가 깨졌거나 HEADER가 없으면 UNKNOWN으로 채움.
    """
    try:
        h = (body or {}).get("HEADER", {}) if isinstance(body, dict) else {}
        return Header(
            IF_ID=h.get("IF_ID", "UNKNOWN"),
            IF_SYSID=h.get("IF_SYSID", "UNKNOWN"),
            IF_HOST=h.get("IF_HOST", "UNKNOWN"),
            IF_DATE=h.get("IF_DATE", _now_yyyymmddhhmmss()),
        )
    except Exception:
        return Header(
            IF_ID="UNKNOWN",
            IF_SYSID="UNKNOWN",
            IF_HOST="UNKNOWN",
            IF_DATE=_now_yyyymmddhhmmss(),
        )

def _validation_error_details(exc: RequestValidationError) -> List[dict]:
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        details.append(
            {
                "location": location or "body",
                "message": str(error.get("msg", "invalid value")),
                "type": str(error.get("type", "validation_error")),
            }
        )
    return details


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    body = request_payload_from_state(request)
    if not body:
        try:
            parsed = await request.json()
            body = parsed if isinstance(parsed, dict) else {}
        except Exception:
            body = {}

    req_header = _safe_req_header(body)
    details = _validation_error_details(exc)
    detail_message = "; ".join(
        f"{item['location']}: {item['message']}" for item in details[:5]
    )
    result_message = f"invalid request: {detail_message}" if detail_message else "invalid request"

    log_event(
        "request_validation_error",
        level="warning",
        method=request.method,
        path=request.url.path,
        request_id=req_header.IF_SYSID,
        status_code=400,
        errors=details,
    )
    return make_error_response(req_header, 400, result_message)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    body = request_payload_from_state(request)
    req_header = _safe_req_header(body)
    detail = (
        exc.detail
        if isinstance(exc.detail, str)
        else json.dumps(exc.detail, ensure_ascii=False)
    )
    log_event(
        "http_exception",
        level="warning" if exc.status_code < 500 else "error",
        method=request.method,
        path=request.url.path,
        request_id=req_header.IF_SYSID,
        status_code=exc.status_code,
        detail=detail,
    )
    return make_error_response(req_header, exc.status_code, detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    body = request_payload_from_state(request)
    req_header = _safe_req_header(body)
    error_id = str(uuid.uuid4())
    error_response = make_response(
        req_header,
        "F",
        f"internal server error (error_id={error_id})",
    )
    started_perf = getattr(request.state, "http_started_perf", None)
    duration_ms = (
        round((time.perf_counter() - started_perf) * 1000, 2)
        if isinstance(started_perf, (int, float))
        else None
    )
    client_ip = request.client.host if request.client else ""
    response_payload = error_response.model_dump()
    log_event(
        "http_transaction",
        level="error",
        request_id=req_header.IF_SYSID,
        method=request.method,
        path=request.url.path,
        query=request.url.query,
        client_ip=client_ip,
        status_code=500,
        duration_ms=duration_ms,
        request_payload=body or None,
        response_payload=response_payload,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
    )
    log_event(
        "unhandled_exception",
        level="error",
        method=request.method,
        path=request.url.path,
        request_id=req_header.IF_SYSID,
        status_code=500,
        error_id=error_id,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        traceback=traceback.format_exc(),
        response_payload=response_payload,
    )
    return JSONResponse(status_code=500, content=response_payload)


# ===== [추가 끝] =====


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------
# 테스트: 어노테이션 job 수동 실행 (자동 트리거 전에 동작 확인용)
# POST /v1/test/run-annotation {"division_idx": "DI17647205538493077", "is_cold": true}
# - MongoDB 조회 후 product_list 순차 처리
# -----------------------------
class TestRunAnnotationBody(BaseModel):
    division_idx: str
    is_cold: bool = True
    only_missing_artifacts: bool = False


class TestAdvancePipelineBody(BaseModel):
    division_idx: str
    is_cold: bool = True
    run_id: str = ""


@app.post("/v1/test/advance-pipeline")
def test_advance_pipeline(body: TestAdvancePipelineBody, background_tasks: BackgroundTasks):
    """테스트/복구용: segment·학습·배포 advance를 수동 트리거 (status=4/6 등)."""
    division_idx = body.division_idx
    is_cold = body.is_cold
    run_id = (body.run_id or "").strip()
    active = find_active_run(_get_mongo_collection, division_idx, is_cold)
    if not run_id and active:
        run_id = str(active.get("run_id") or "")
    if active and active.get("log_dir"):
        request_ctx = resume_request_log(
            active,
            event="test_advance_pipeline",
            extra={"run_id": run_id or None},
        )
    else:
        request_ctx = init_request_log(
            event="test_advance_pipeline",
            division_idx=division_idx,
            is_cold=is_cold,
        )
    background_tasks.add_task(
        _maybe_advance_pipeline,
        division_idx,
        is_cold,
        request_ctx,
        run_id,
    )
    return {
        "started": True,
        "division_idx": division_idx,
        "is_cold": is_cold,
        "run_id": run_id,
        "message": "pipeline advance를 백그라운드에서 실행합니다.",
    }


@app.post("/v1/test/run-annotation")
def test_run_annotation(body: TestRunAnnotationBody, background_tasks: BackgroundTasks):
    """테스트용: product/created와 동일한 흐름으로 어노테이션 job을 수동 실행합니다."""
    division_idx = body.division_idx
    is_cold = body.is_cold
    product_list = _resolve_product_info_from_mongo(division_idx, is_cold=is_cold)
    if not product_list:
        return {"started": False, "error": "division_idx lookup failed (MongoDB)"}

    active = find_active_run(_get_mongo_collection, division_idx, is_cold)
    run_id = str((active or {}).get("run_id") or "")
    request_ctx = None
    if active and active.get("log_dir"):
        request_ctx = resume_request_log(
            active,
            event="test_run_annotation",
            extra={"only_missing_artifacts": body.only_missing_artifacts},
        )

    reconciled_count = 0
    if body.only_missing_artifacts:
        candidates = [
            product
            for product in product_list
            if str(product.get("training_status") or "").strip() == "2"
            and _is_new_product(product.get("is_new"))
        ]
        product_list = []
        for product in candidates:
            if has_new_annotation_artifact(product):
                product_object_id = str(product.get("product_object_id") or "")
                if run_id and product_object_id:
                    update_product_phase(
                        _get_mongo_collection,
                        run_id,
                        product_object_id,
                        PHASE_AWAITING_REVIEW,
                    )
                reconciled_count += 1
                continue
            product_list.append(product)

    if not product_list:
        return {
            "started": False,
            "division_idx": division_idx,
            "is_cold": is_cold,
            "run_id": run_id,
            "product_count": 0,
            "reconciled_count": reconciled_count,
            "message": "실행할 미완료 자동 어노테이션 상품이 없습니다.",
        }

    _schedule_annotation_batch(
        background_tasks,
        division_idx,
        product_list,
        is_cold,
        request_ctx,
        run_id,
    )
    return {
        "started": True,
        "division_idx": division_idx,
        "is_cold": is_cold,
        "run_id": run_id,
        "product_count": len(product_list),
        "reconciled_count": reconciled_count,
        "train_product_idxs": [product.get("train_product_idx") for product in product_list],
        "message": "어노테이션 job을 백그라운드에서 순차 실행합니다. 서버 로그에서 진행 상황을 확인하세요.",
    }

# -----------------------------
# A) Edge -> AI : 신규상품 등록 트리거
# 최소: division_idx
# -----------------------------
class ProductCreatedData(BaseModel):
    division_idx: str
    is_cold: bool

class ProductCreatedEnvelope(BaseModel):
    HEADER: Header
    DATA: ProductCreatedData


class MongoConfigurationError(RuntimeError):
    pass


class MongoQueryError(RuntimeError):
    pass


def _get_mongo_collection(name: str):
    global _mongo_client
    mongo_uri = os.getenv("MONGO_URI", "").strip()
    mongo_db = os.getenv("MONGO_DB", "").strip()
    if not mongo_uri or not mongo_db:
        print("[mongo] missing config: MONGO_URI or MONGO_DB is empty")
        return None
    if _mongo_client is None:
        _mongo_client = MongoClient(mongo_uri)
    return _mongo_client[mongo_db][name]


def _find_device_doc(device_col, division_idx: str, storage_type: str):
    """Prefer the requested storageType, then fall back to UNKNOWN for legacy documents."""
    candidates = [storage_type]
    if storage_type != "UNKNOWN":
        candidates.append("UNKNOWN")

    for candidate in candidates:
        device_doc = device_col.find_one(
            {"divisionIdx": division_idx, "storageType": candidate},
            sort=[("trainingDate", -1), ("_id", -1)],
        )
        if device_doc:
            if candidate != storage_type:
                print(
                    f"[mongo] DeviceTypeList fallback 조회 성공: "
                    f"divisionIdx={division_idx}, requested={storage_type}, actual={candidate}"
                )
            return device_doc
    return None


def _resolve_product_info_from_mongo(
    division_idx: str,
    is_cold: bool = False,
    raise_on_error: bool = False,
) -> list:
    """
    division_idx + is_cold로 어노테이션에 필요한 정보를 조회합니다.
    DeviceTypeList에서 storageType(COLD/FROZEN) + divisionIdx로 조회합니다.

    DB 구조:
      - DivisionList: divisionIdx, deviceIdx[] (장비 index만, products 없음)
      - DeviceTypeList: divisionIdx, storageType, brunchName, deviceIdx[], products[]

    흐름:
      1) DeviceTypeList 조회 (divisionIdx + storageType)
         → products[] 전체 순회, brunchName 획득
      2) 각 product ObjectId로 ProductsList 조회 → foldername, folderpath 획득

    반환: product_info dict 리스트 (각 항목에 brunch_name 포함)
    """
    device_type_collection = os.getenv("MONGO_DEVICETYPELIST_COLLECTION", "DeviceTypeList")
    products_collection_name = os.getenv("MONGO_PRODUCTSLIST_COLLECTION", "ProductsList")
    # is_cold=True → COLD(냉동), is_cold=False → FROZEN(냉장) - 환경변수로 오버라이드 가능
    storage_type = "COLD" if is_cold else os.getenv("STORAGE_TYPE_NOT_COLD", "FROZEN")

    try:
        device_col = _get_mongo_collection(device_type_collection)
        if device_col is None:
            if raise_on_error:
                raise MongoConfigurationError("MONGO_URI or MONGO_DB is not configured")
            return []

        device_doc = _find_device_doc(device_col, division_idx, storage_type)
        if not device_doc:
            print(
                f"[mongo] DeviceTypeList 문서 없음: divisionIdx={division_idx}, "
                f"storageType={storage_type} (including UNKNOWN fallback)"
            )
            return []

        brunch_name = device_doc.get("brunchName")
        print(f"[mongo] DeviceTypeList 조회 성공: brunchName={brunch_name}")

        from bson import ObjectId
        products_col = _get_mongo_collection(products_collection_name)
        if products_col is None:
            if raise_on_error:
                raise MongoConfigurationError("MONGO_URI or MONGO_DB is not configured")
            return []

        product_list = []
        for item in device_doc.get("products", []):
            if not isinstance(item, dict) or not item.get("product"):
                continue
            product_object_id = item["product"]
            training_status = item.get("training_status")

            product_doc = products_col.find_one({"_id": ObjectId(str(product_object_id))})
            if not product_doc:
                print(f"[mongo] ProductsList 문서 없음: _id={product_object_id}")
                continue

            product_list.append({
                "product_idx": product_doc.get("productIdx"),
                "product_object_id": str(product_object_id),
                "brunch_name": brunch_name,
                "foldername": product_doc.get("foldername"),
                "folderpath": product_doc.get("folderpath"),
                "train_product_idx": product_doc.get("trainProductIdx"),
                "training_status": training_status,
                "product_eng_name": product_doc.get("productEngName"),
                "is_new": product_doc.get("isNew"),
            })

        print(f"[mongo] ProductsList 조회 완료: {len(product_list)}개 product")
        return product_list

    except PyMongoError as e:
        print(f"[mongo] 조회 실패: {e}")
        if raise_on_error:
            raise MongoQueryError(str(e)) from e
        return []


def _build_product_info(product_doc: dict, brunch_name: str, training_status=None) -> dict:
    return {
        "product_idx": product_doc.get("productIdx"),
        "product_object_id": str(product_doc.get("_id")),
        "brunch_name": brunch_name,
        "foldername": product_doc.get("foldername"),
        "folderpath": product_doc.get("folderpath"),
        "train_product_idx": product_doc.get("trainProductIdx"),
        "training_status": training_status,
        "product_eng_name": product_doc.get("productEngName"),
        "is_new": product_doc.get("isNew"),
    }


def _is_new_product(is_new_value) -> bool:
    """
    ProductsList.isNew 판별.
    "1" → 신규(어노테이션·세그먼트 필요), "0" → 기학습 상품(학습 직전 status로 바로 진행).
    """
    if is_new_value is None:
        return True
    return str(is_new_value).strip() != "0"


def _has_product_seg_pkl(product_info: dict) -> bool:
    train_product_idx = str(product_info.get("train_product_idx") or "").strip()
    product_eng_name = str(product_info.get("product_eng_name") or "").strip()
    if not train_product_idx or not product_eng_name:
        return False

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
    secure = os.getenv("MINIO_SECURE", "false").strip().lower() == "true"
    prefix = f"productSeg/{train_product_idx}_{product_eng_name}_"
    try:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            if obj.object_name.endswith(".pkl"):
                return True
    except Exception as e:
        pipeline_log(
            f"productSeg 확인 실패: prefix={prefix}, error={e}",
            phase="is_new",
            level="ERROR",
        )
    return False


def _is_effectively_new_product(product_info: dict) -> bool:
    if not _is_new_product(product_info.get("is_new")):
        return False
    if _has_product_seg_pkl(product_info):
        pipeline_log(
            "isNew=1 이지만 productSeg pkl 존재 → 기존 상품으로 처리: "
            f"{product_info.get('foldername')}",
            phase="is_new",
        )
        return False
    return True


def _split_products_by_is_new(products: list) -> Tuple[list, list]:
    """Returns (effective new products, existing/segment-ready products)."""
    new_products = []
    existing_products = []
    for product in products:
        if _is_effectively_new_product(product):
            new_products.append(product)
        else:
            existing_products.append(product)
    return new_products, existing_products


def _update_is_new_in_products_list(product_object_id: str, is_new: str = "0") -> bool:
    """ProductsList.isNew 갱신 (학습 완료 후 0으로)."""
    from bson import ObjectId

    col = _get_mongo_collection(os.getenv("MONGO_PRODUCTSLIST_COLLECTION", "ProductsList"))
    if col is None:
        return False
    try:
        result = col.update_one(
            {"_id": ObjectId(product_object_id)},
            {"$set": {"isNew": str(is_new)}},
        )
        return result.matched_count > 0
    except PyMongoError as e:
        print(f"[mongo] ProductsList isNew 업데이트 오류: {e}")
        return False


def _bind_request_ctx(request_ctx: Optional[RequestLogContext]) -> None:
    if request_ctx:
        bind_request_context(request_ctx)


def _run_id_from_ctx(request_ctx: Optional[RequestLogContext]) -> str:
    if not request_ctx:
        return ""
    return (request_ctx.if_sysid or request_ctx.request_id or "").strip()

def _update_training_status_in_mongo(brunch_name: str, product_object_id: str, status: str = "2") -> bool:
    """
    DeviceTypeList.products[].training_status를 MongoDB에서 직접 업데이트합니다.
    어노테이션 완료 시 status="2" 등으로 호출합니다.
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        col = _get_mongo_collection(os.getenv("MONGO_DEVICETYPELIST_COLLECTION", "DeviceTypeList"))
        if col is None:
            return False
        object_id = ObjectId(product_object_id)
    except (InvalidId, MongoConfigurationError, PyMongoError) as e:
        print(f"[mongo] training_status 업데이트 준비 실패: product={product_object_id}, {e}")
        return False
    try:
        result = col.update_one(
            {"brunchName": brunch_name, "products.product": object_id},
            {"$set": {"products.$.training_status": status}},
        )
        if result.matched_count > 0:
            if result.modified_count > 0:
                pipeline_log(
                    f"Mongo training_status → {status}: brunchName={brunch_name}, product={product_object_id}",
                    phase="mongo",
                )
                if not _post_training_status_if07(brunch_name, product_object_id, status):
                    pipeline_log(
                        f"IF_07 POST 실패(상태는 Mongo에 반영됨): brunchName={brunch_name}, "
                        f"product={product_object_id}, status={status}",
                        phase="IF_07",
                        level="ERROR",
                    )
                else:
                    pipeline_log(
                        f"IF_07 POST 성공: product={product_object_id}, status={status}",
                        phase="IF_07",
                    )
            else:
                pipeline_log(
                    f"training_status 이미 {status} (변경 없음): brunchName={brunch_name}, "
                    f"product={product_object_id}",
                    phase="mongo",
                )
            return True
        print(f"[mongo] training_status 업데이트 실패 (매칭 문서 없음): brunchName={brunch_name}")
        return False
    except PyMongoError as e:
        print(f"[mongo] training_status 업데이트 오류: {e}")
        return False


def _is_jwt_expired(token: str) -> bool:
    """JWT exp 필드 확인 — 만료됐거나 파싱 실패 시 True."""
    try:
        import base64 as _b64, json as _json, time as _time
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        decoded = _json.loads(_b64.b64decode(payload))
        exp = decoded.get("exp")
        if exp is None:
            return False  # exp 없으면 만료 없는 토큰으로 간주
        return _time.time() > float(exp)
    except Exception:
        return False


def _get_pnt_access_token() -> str:
    token = os.getenv("PNT_ACCESS_TOKEN", "").strip()
    if token and not _is_jwt_expired(token):
        return token
    if token and _is_jwt_expired(token):
        print("[pnt] 토큰 만료 — 재로그인 시도")
    try:
        try:
            from .pnt_login import get_token
        except ImportError:
            from pnt_login import get_token
        new_token = (get_token() or "").strip()
        if new_token:
            os.environ["PNT_ACCESS_TOKEN"] = new_token
            # .env 파일도 업데이트
            try:
                env_path = os.path.join(os.path.dirname(__file__), ".env")
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        env_content = f.read()
                    import re as _re
                    env_content = _re.sub(
                        r"PNT_ACCESS_TOKEN=.*",
                        f"PNT_ACCESS_TOKEN={new_token}",
                        env_content,
                    )
                    with open(env_path, "w") as f:
                        f.write(env_content)
                    print("[pnt] 새 토큰으로 .env 업데이트 완료")
            except Exception as e:
                print(f"[pnt] .env 업데이트 실패 (메모리 토큰은 갱신됨): {e}")
            return new_token
        return token  # 재로그인 실패 시 기존 토큰이라도 반환
    except Exception as e:
        print(f"[pnt] 토큰 조회 실패: {e}")
        return token


def _post_pnt_json(if_id: str, path: str, payload: dict) -> bool:
    base = os.getenv("PNT_URL", "https://apichaidev.atcrk.co.kr").rstrip("/")
    url = f"{base}{path}"
    token = _get_pnt_access_token()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": base,
        "Referer": f"{base}/",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = float(os.getenv("PNT_POST_TIMEOUT", "10"))
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            status_code = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="ignore")
            pipeline_log(f"POST {url} status={status_code}", phase=if_id)
            if status_code != 200:
                pipeline_log(f"응답 오류: {body[:500]}", phase=if_id, level="ERROR")
                return False
            if body:
                try:
                    parsed = json.loads(body)
                    result_cd = str(parsed.get("DATA", {}).get("result_cd", ""))
                    if result_cd and result_cd != "S":
                        pipeline_log(f"result_cd 실패: {body[:500]}", phase=if_id, level="ERROR")
                        return False
                except Exception:
                    pass
            return True
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
        pipeline_log(f"HTTPError {e.code}: {body[:500]}", phase=if_id, level="ERROR")
        return False
    except URLError as e:
        pipeline_log(f"URLError: {e}", phase=if_id, level="ERROR")
        return False
    except Exception as e:
        pipeline_log(f"예외: {e}", phase=if_id, level="ERROR")
        return False


def _resolve_if07_product_payload(brunch_name: str, product_object_id: str, status: str) -> Optional[dict]:
    from bson import ObjectId

    def _normalize_device_idx(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            for v in value:
                s = str(v).strip()
                if s:
                    return s
            return ""
        return str(value).strip()

    products_col = _get_mongo_collection(os.getenv("MONGO_PRODUCTSLIST_COLLECTION", "ProductsList"))
    device_col = _get_mongo_collection(os.getenv("MONGO_DEVICETYPELIST_COLLECTION", "DeviceTypeList"))
    if products_col is None or device_col is None:
        return None

    product_doc = products_col.find_one({"_id": ObjectId(product_object_id)})
    if not product_doc:
        print(f"[pnt][IF_07] ProductsList 문서 없음: {product_object_id}")
        return None

    device_doc = device_col.find_one({"brunchName": brunch_name, "products.product": ObjectId(product_object_id)})
    division_idx = brunch_name.rsplit("_", 1)[0] if "_" in brunch_name else brunch_name
    device_idx = ""
    if device_doc:
        division_idx = str(device_doc.get("divisionIdx") or division_idx)
        device_idx = _normalize_device_idx(device_doc.get("deviceIdx"))
        if not device_idx:
            device_idx = _normalize_device_idx(device_doc.get("device_idx"))

    return {
        "division_idx": str(division_idx),
        "device_idx": device_idx,
        "product_idx": str(product_doc.get("productIdx") or ""),
        "product_eng_name": str(product_doc.get("productEngName") or product_doc.get("foldername") or ""),
        "training_status": str(status),
    }


def _post_training_status_if07(brunch_name: str, product_object_id: str, status: str) -> bool:
    item = _resolve_if07_product_payload(brunch_name, product_object_id, status)
    if item is None:
        return False
    payload = {
        "HEADER": {
            "IF_ID": "IF_07",
            "IF_SYSID": str(uuid.uuid4()),
            "IF_HOST": os.getenv("IF_HOST", socket.gethostname()),
            "IF_DATE": datetime.now().strftime("%Y%m%d%H%M%S"),
        },
        "DATA": {
            "product_list": [item],
        },
    }
    path = os.getenv("PNT_PATH_IF07", os.getenv("PNT_PATH", "/api/v1/chai/training/store"))
    return _post_pnt_json("IF_07", path, payload)


def _classify_products_by_status(all_products: list) -> Tuple[list, list, list, list]:
    """
    training_status별 product 분류.
    Returns: (status_2, status_3, status_4_plus, status_6)
    training_status가 None인 product는 분류에서 제외됨.
    """
    products_status_2 = []
    products_status_3 = []
    products_status_4_plus = []
    products_status_6 = []

    for product in all_products:
        status = product.get("training_status")
        if status is None:
            continue

        try:
            status_str = str(status)
            status_int = int(status_str) if status_str.isdigit() else None

            if status_str == "2":
                products_status_2.append(product)
            elif status_str == "3":
                products_status_3.append(product)
            elif status_str == "6":
                products_status_6.append(product)
            elif status_int is not None and status_int >= 4:
                products_status_4_plus.append(product)
        except (ValueError, TypeError):
            print(f"[product_created] training_status 파싱 실패: {status}, product_idx={product.get('product_idx')}")
            continue

    return products_status_2, products_status_3, products_status_4_plus, products_status_6


def _pipeline_lock_key(division_idx: str, is_cold: bool) -> str:
    return f"{division_idx}:{'C' if is_cold else 'F'}"


def _get_pipeline_lock(division_idx: str, is_cold: bool) -> threading.Lock:
    key = _pipeline_lock_key(division_idx, is_cold)
    if key not in _pipeline_locks:
        _pipeline_locks[key] = threading.Lock()
    return _pipeline_locks[key]


def _annotation_lock_key(division_idx: str, is_cold: bool) -> str:
    return f"{division_idx}:{'C' if is_cold else 'F'}"


def _get_annotation_lock(division_idx: str, is_cold: bool) -> threading.Lock:
    key = _annotation_lock_key(division_idx, is_cold)
    if key not in _annotation_locks:
        _annotation_locks[key] = threading.Lock()
    return _annotation_locks[key]


def _claim_annotation_batch(division_idx: str, is_cold: bool, run_id: str) -> bool:
    """
    division당 어노테이션 배치 1개. 새 product/created가 오면 active run_id를 갱신하고
    이전 배치는 다음 상품 처리 전에 중단된다.
    """
    key = _annotation_lock_key(division_idx, is_cold)
    _annotation_active_run[key] = run_id
    lock = _get_annotation_lock(division_idx, is_cold)
    wait_sec = int(os.getenv("ANNOTATION_LOCK_WAIT_SEC", "300"))
    acquired = lock.acquire(blocking=True, timeout=wait_sec)
    if not acquired:
        pipeline_log(
            f"annotation batch lock timeout ({wait_sec}s): division_idx={division_idx}, run_id={run_id}",
            phase="annotation",
            level="ERROR",
        )
        return False
    if _annotation_active_run.get(key) != run_id:
        lock.release()
        pipeline_log(
            f"annotation batch superseded before start: division_idx={division_idx}, run_id={run_id}",
            phase="annotation",
        )
        return False
    return True


def _annotation_batch_cancelled(division_idx: str, is_cold: bool, run_id: str) -> bool:
    key = _annotation_lock_key(division_idx, is_cold)
    return _annotation_active_run.get(key) != run_id


def _release_annotation_batch(division_idx: str, is_cold: bool, run_id: str) -> None:
    key = _annotation_lock_key(division_idx, is_cold)
    lock = _get_annotation_lock(division_idx, is_cold)
    if _annotation_active_run.get(key) == run_id:
        _annotation_active_run.pop(key, None)
    try:
        if lock.locked():
            lock.release()
    except RuntimeError:
        pass


def _run_background_task_isolated(func, *args, **kwargs) -> None:
    """FastAPI는 background task가 예외를 내면 같은 요청의 남은 task를 실행하지 않는다.

    어노테이션 drain worker가 선행 task의 실패로 시작조차 못 하면 해당 division 큐가
    영구히 멈추므로, task 하나의 실패를 여기서 흡수한다.
    """
    name = getattr(func, "__name__", str(func))
    try:
        func(*args, **kwargs)
    except Exception as exc:
        try:
            pipeline_log(
                f"background task 실패(격리됨): {name} — {type(exc).__name__}: {exc}",
                phase="pipeline",
                level="ERROR",
            )
        except Exception:
            print(f"[pipeline] background task 실패(격리됨): {name} — {exc}", flush=True)


def _add_background_task(background_tasks: BackgroundTasks, func, *args, **kwargs) -> None:
    background_tasks.add_task(_run_background_task_isolated, func, *args, **kwargs)


def _drain_annotation_batches(queue_key: str, token: int) -> None:
    """Run queued batches serially for one division while preserving each request context."""
    try:
        while True:
            batch = annotation_batch_queue.pop(queue_key, token)
            if batch is None:
                return
            division_idx, product_list, is_cold, request_ctx, run_id = batch
            try:
                _run_annotation_after_created(
                    division_idx,
                    product_list,
                    is_cold,
                    request_ctx,
                    run_id,
                )
            except Exception as exc:
                _bind_request_ctx(request_ctx)
                pipeline_log(
                    f"annotation queue worker 예외: run_id={run_id}, "
                    f"{type(exc).__name__}: {exc}",
                    phase="annotation",
                    level="ERROR",
                )
    finally:
        # 워커가 비정상 종료해도 key를 반납해야 다음 요청이 새 워커를 띄울 수 있다.
        # 남은 batch는 큐에 그대로 있으므로 다음 워커가 순서대로 이어서 처리한다.
        if annotation_batch_queue.retire(queue_key, token):
            leftover = annotation_batch_queue.pending_count(queue_key)
            if leftover:
                pipeline_log(
                    f"annotation queue worker 비정상 종료 — 남은 batch {leftover}개는 "
                    f"다음 요청이 이어서 처리: queue_key={queue_key}",
                    phase="annotation",
                    level="ERROR",
                )


def _schedule_annotation_batch(
    background_tasks: BackgroundTasks,
    division_idx: str,
    product_list: list,
    is_cold: bool,
    request_ctx: Optional[RequestLogContext],
    run_id: str,
) -> None:
    """Append work to the division queue and start exactly one drain worker."""
    unique_products = []
    seen_products = set()
    for product in product_list:
        identity = str(
            product.get("product_object_id")
            or product.get("folderpath")
            or product.get("foldername")
            or ""
        )
        if identity and identity in seen_products:
            continue
        unique_products.append(product)
        if identity:
            seen_products.add(identity)
    if not unique_products:
        return

    queue_key = _annotation_lock_key(division_idx, is_cold)
    worker_token, queued_count = annotation_batch_queue.enqueue(
        queue_key,
        (division_idx, unique_products, is_cold, request_ctx, run_id),
    )
    pipeline_log(
        f"annotation batch 큐 등록: run_id={run_id}, products={len(unique_products)}, "
        f"queued_batches={queued_count}, "
        f"worker={'start' if worker_token is not None else 'running'}",
        phase="annotation",
    )
    if worker_token is not None:
        _add_background_task(background_tasks, _drain_annotation_batches, queue_key, worker_token)


def _annotation_gpu_ids() -> List[int]:
    raw = os.getenv("PIPELINE_ANNOTATION_GPUS", "0,1")
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids[:2] if ids else [0]


def _annotation_gpu_slot_wait_sec() -> int:
    default_wait_sec = 14400
    raw = os.getenv("ANNOTATION_GPU_SLOT_WAIT_SEC", str(default_wait_sec))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        pipeline_log(
            f"ANNOTATION_GPU_SLOT_WAIT_SEC 값이 잘못되어 기본값 {default_wait_sec}초 사용: {raw!r}",
            phase="annotation",
            level="ERROR",
        )
        return default_wait_sec
    return max(1, value)


def _maybe_advance_pipeline(
    division_idx: str,
    is_cold: bool = False,
    request_ctx: Optional[RequestLogContext] = None,
    run_id: str = "",
) -> None:
    """
    job 완료 후 MongoDB를 다시 읽어 division 전체가 준비되면
    학습 → (검증 통과 시) 배포까지 자동 진행.
    같은 락 안에서 상태를 재조회하며 연속 advance (학습 후 배포 skip 방지).
    """
    lock = _get_pipeline_lock(division_idx, is_cold)
    if not lock.acquire(blocking=False):
        pipeline_log(f"skip (다른 advance 진행 중): division_idx={division_idx}", phase="pipeline")
        return
    try:
        _bind_request_ctx(request_ctx)
        for _step in range(4):
            all_products = _resolve_product_info_from_mongo(division_idx, is_cold=is_cold)
            if not all_products:
                return

            statuses = [
                str(p.get("training_status"))
                for p in all_products
                if p.get("training_status") is not None
            ]
            if not statuses:
                return

            if any(s in ("2", "3") for s in statuses):
                n2 = sum(1 for s in statuses if s == "2")
                n3 = sum(1 for s in statuses if s == "3")
                pipeline_log(
                    f"대기: division_idx={division_idx}, status=2({n2}), status=3({n3})",
                    phase="pipeline",
                )
                return

            if any(s == "8" for s in statuses):
                pipeline_log(
                    f"검증 미통과(status=8) — 수동 확인 필요: division_idx={division_idx}",
                    phase="pipeline",
                    level="ERROR",
                )
                return

            if all(s == "7" for s in statuses):
                pipeline_log(f"배포 완료(status=7): division_idx={division_idx}", phase="pipeline")
                return

            if all(s == "6" for s in statuses):
                pipeline_log(
                    f"division 전체 status=6 → 배포 시작: division_idx={division_idx}",
                    phase="pipeline",
                )
                if run_id:
                    update_run_status(_get_mongo_collection, run_id, RUN_DEPLOYED)
                _run_deploy_job(division_idx, all_products, is_cold, request_ctx=request_ctx, run_id=run_id)
                return

            if all(s == "4" for s in statuses):
                pipeline_log(
                    f"division 전체 status=4 → 학습 시작: division_idx={division_idx}",
                    phase="pipeline",
                )
                if run_id:
                    update_run_status(_get_mongo_collection, run_id, RUN_TRAINING)
                trained = _run_training_job(division_idx, all_products, is_cold, request_ctx=request_ctx, run_id=run_id)
                if not trained:
                    return
                continue

            pipeline_log(
                f"advance 조건 미충족: division_idx={division_idx}, statuses={statuses}",
                phase="pipeline",
            )
            return
    finally:
        lock.release()


def post_model_deploy_if14(division_idx: str, device_idx: str = "", model_version: str = "", status: str = "7", is_cold: bool = False, brunch_name: str = None) -> bool:
    """
    배포 완료 시 IF_14로 status=7 전송 및 MongoDB 업데이트.
    
    Args:
        division_idx: division index
        device_idx: device index
        model_version: 모델 버전 (예: "v26.5.21")
        status: training status (기본값: "7")
        is_cold: 냉동 여부
        brunch_name: brunch name (없으면 자동 조회)
    
    Returns:
        성공 여부
    """
    def _normalize_device_idx_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        s = str(value).strip()
        return [s] if s else []

    storage_type = "COLD" if is_cold else os.getenv("STORAGE_TYPE_NOT_COLD", "FROZEN")
    device_type_collection = os.getenv("MONGO_DEVICETYPELIST_COLLECTION", "DeviceTypeList")
    device_col = _get_mongo_collection(device_type_collection)
    device_doc = None
    device_idx_list = _normalize_device_idx_list(device_idx)

    if device_col is not None:
        try:
            device_doc = _find_device_doc(device_col, division_idx, storage_type)
            if device_doc:
                if not brunch_name:
                    brunch_name = device_doc.get("brunchName")
                if not device_idx_list:
                    device_idx_list = _normalize_device_idx_list(device_doc.get("deviceIdx"))
                if not device_idx_list:
                    device_idx_list = _normalize_device_idx_list(device_doc.get("device_idx"))
                if brunch_name:
                    for item in device_doc.get("products", []):
                        if isinstance(item, dict) and item.get("product"):
                            product_object_id = str(item["product"])
                            _update_training_status_in_mongo(brunch_name, product_object_id, status=status)
                    print(f"[deploy] MongoDB status 업데이트 완료: division_idx={division_idx}, brunch_name={brunch_name}, status={status}")
        except Exception as e:
            print(f"[deploy] MongoDB 업데이트 오류: {e}")

    now_iso = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    payload = {
        "HEADER": {
            "IF_ID": "IF_14",
            "IF_SYSID": str(uuid.uuid4()),
            "IF_HOST": os.getenv("IF_HOST", socket.gethostname()),
            "IF_DATE": datetime.now().strftime("%Y%m%d%H%M%S"),
        },
        "DATA": {
            "device_list": [
                {
                    "division_idx": str(division_idx),
                    "device_idx": str(dev_idx),
                    "brunch_name": str(brunch_name or ""),
                    "brunch_update_date": now_iso,
                    "model_version": str(model_version),
                    "model_update_date": now_iso,
                }
                for dev_idx in device_idx_list
            ]
        },
    }
    path = os.getenv("PNT_PATH_IF14", "/api/v1/chai/device/store")
    return _post_pnt_json("IF_14", path, payload)


def _run_segment_job(
    division_idx: str,
    product_info: dict,
    is_cold: bool = False,
    request_ctx: Optional[RequestLogContext] = None,
    run_id: str = "",
) -> None:
    """어노테이션과 공유하는 GPU 슬롯을 예약한 뒤 세그먼트를 실행."""
    _bind_request_ctx(request_ctx)
    if not run_id:
        run_id = _run_id_from_ctx(request_ctx)
    brunch_name = product_info.get("brunch_name") or f"{division_idx}_{'C' if is_cold else 'F'}"
    foldername = product_info.get("foldername", "")
    obj_id = product_info.get("product_object_id")
    if run_id and obj_id:
        update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_PENDING_SEGMENT)
    gpu_pool = None
    gpu_id = None
    slot_label = f"segment:{foldername}"
    ok = False
    try:
        gpu_pool = get_gpu_slot_pool(
            _annotation_gpu_ids(), _annotation_gpu_slot_wait_sec(), logger=pipeline_log
        )
        gpu_id = gpu_pool.acquire(slot_label)
        if run_id and obj_id:
            update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_SEGMENTING)
        pipeline_log(
            f"segment 시작: product_idx={product_info.get('product_idx')}, "
            f"foldername={foldername}, GPU {gpu_id}",
            phase="segment",
        )
        ok = run_segment_job(division_idx, product_info, is_cold=is_cold, gpu_id=gpu_id)
    except Exception as exc:
        pipeline_log(
            f"segment 실행 실패: {foldername} — {type(exc).__name__}: {exc}",
            phase="segment",
            level="ERROR",
        )
    finally:
        if gpu_pool is not None and gpu_id is not None:
            gpu_pool.release(gpu_id, slot_label)

    if ok:
        if obj_id:
            ok = _update_training_status_in_mongo(brunch_name, obj_id, status="4")
            if ok:
                if run_id:
                    update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_READY)
                pipeline_log(f"segment 완료, training_status → 4: {foldername}", phase="segment")
    if not ok:
        if run_id and obj_id:
            update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_FAILED)
        pipeline_log(f"segment 실패: {foldername}", phase="segment", level="ERROR")

    active = find_active_run(_get_mongo_collection, division_idx, is_cold)
    pipeline_log(run_summary(active), phase="pipeline_run")
    _maybe_advance_pipeline(division_idx, is_cold, request_ctx=request_ctx, run_id=run_id)


def _run_training_job(
    division_idx: str,
    product_list: list,
    is_cold: bool = False,
    request_ctx: Optional[RequestLogContext] = None,
    run_id: str = "",
) -> bool:
    """학습 job 실행. 성공 시 True, 실패 시 False 반환."""
    _bind_request_ctx(request_ctx)
    pipeline_log(f"training 시작: division_idx={division_idx}", phase="training")
    result = run_training_job(division_idx=division_idx, product_list=product_list, is_cold=is_cold)
    if not result.get("train_success"):
        print(f"[training] 학습 실패: division_idx={division_idx}")
        if run_id:
            update_run_status(_get_mongo_collection, run_id, RUN_FAILED)
        return False

    val_score = result.get("validation_score")
    val_passed = bool(result.get("validation_passed"))

    # 학습 완료 step
    for product_info in product_list:
        brunch_name = product_info.get("brunch_name") or f"{division_idx}_{'C' if is_cold else 'F'}"
        obj_id = product_info.get("product_object_id")
        if obj_id:
            _update_training_status_in_mongo(brunch_name, obj_id, status="5")
    print(f"[training] 학습 완료 및 training_status 업데이트: status=5, val_score={val_score}")

    # 검증 실패 시 8,  통과 시 다음 step(6) 진행
    next_status = "6" if val_passed else "8"
    for product_info in product_list:
        brunch_name = product_info.get("brunch_name") or f"{division_idx}_{'C' if is_cold else 'F'}"
        obj_id = product_info.get("product_object_id")
        if obj_id:
            _update_training_status_in_mongo(brunch_name, obj_id, status=next_status)

    if val_passed:
        print(f"[validation] 통과: val_score={val_score}, training_status=6")
    else:
        print(f"[validation] 미통과: val_score={val_score}, training_status=8")

    if val_passed:
        for product_info in product_list:
            obj_id = product_info.get("product_object_id")
            foldername = product_info.get("foldername", "")
            if not obj_id:
                continue
            if _update_is_new_in_products_list(obj_id, "0"):
                pipeline_log(
                    f"ProductsList isNew → 0 (validation 통과 후 학습 사용 완료): {foldername}",
                    phase="is_new",
                )
            else:
                pipeline_log(
                    f"ProductsList isNew → 0 실패: {foldername}",
                    phase="is_new",
                    level="ERROR",
                )
    else:
        pipeline_log(
            "validation 미통과 — ProductsList isNew 유지",
            phase="is_new",
        )
    return True


def _run_deploy_job(
    division_idx: str,
    product_list: list,
    is_cold: bool = False,
    request_ctx: Optional[RequestLogContext] = None,
    run_id: str = "",
) -> None:
    """모델 배포 job 실행."""
    _bind_request_ctx(request_ctx)
    if not product_list:
        pipeline_log(f"product_list가 비어있습니다: division_idx={division_idx}", phase="deploy", level="ERROR")
        return

    brunch_name = product_list[0].get("brunch_name") or f"{division_idx}_{'C' if is_cold else 'F'}"
    pipeline_log(
        f"deploy 시작: division_idx={division_idx}, brunch_name={brunch_name}",
        phase="deploy",
    )
    # MinIO에 모델 업로드
    ok, model_version, model_filename = run_deploy_job(division_idx, brunch_name, is_cold=is_cold)
    if not ok:
        print(f"[deploy] 배포 실패: division_idx={division_idx}")
        return
    
    print(f"[deploy] MinIO 업로드 완료: {brunch_name}/{model_filename}, version={model_version}")
    
    # device_idx 추출 (첫 번째 product에서)
    device_idx = ""
    if product_list:
        # 필요시 device_idx를 product_list에서 추출할 수 있음
        pass
    
    # post_model_deploy_if14에서 MongoDB 업데이트 및 IF_14 전송 모두 처리
    if14_ok = post_model_deploy_if14(
        division_idx,
        device_idx,
        model_version,
        status="7",
        is_cold=is_cold,
        brunch_name=brunch_name,
    )
    if not if14_ok:
        pipeline_log(
            f"IF_14 실패: division_idx={division_idx}, brunch_name={brunch_name}, model_version={model_version}",
            phase="deploy",
            level="ERROR",
        )
        if run_id:
            update_run_status(_get_mongo_collection, run_id, RUN_FAILED)
        return
    print(f"[deploy] 배포 완료 및 training_status 업데이트: status=7")
    if run_id:
        update_run_status(_get_mongo_collection, run_id, RUN_COMPLETED)
    active = find_active_run(_get_mongo_collection, division_idx, is_cold)
    pipeline_log(run_summary(active), phase="pipeline_run")


def _run_annotation_after_created(
    division_idx: str,
    product_list: list,
    is_cold: bool = False,
    request_ctx: Optional[RequestLogContext] = None,
    run_id: str = "",
) -> None:
    """
    NVML 상태를 기준으로 상품별 GPU를 선택하고 카메라를 순차 처리한다.
    조건을 만족하는 GPU가 없으면 프로세스 전역 슬롯 풀에서 대기한다.
    """
    _bind_request_ctx(request_ctx)
    if not run_id:
        run_id = _run_id_from_ctx(request_ctx)
    if not _claim_annotation_batch(division_idx, is_cold, run_id):
        return

    try:
        reset_n = reset_failed_products_to_pending(_get_mongo_collection, run_id)
        if reset_n:
            pipeline_log(
                f"failed → pending_annotation {reset_n}개 (재시도): run_id={run_id}",
                phase="annotation",
            )

        work: List[tuple] = []
        for i, product_info in enumerate(product_list):
            foldername = product_info.get("foldername", "")
            obj_id = product_info.get("product_object_id")
            if obj_id and should_skip_annotation(
                _get_mongo_collection, division_idx, is_cold, obj_id, run_id=run_id
            ):
                pipeline_log(
                    f"skip ({i + 1}/{len(product_list)}): {foldername} — phase skip 또는 stale run",
                    phase="annotation",
                )
                continue
            work.append((i + 1, len(product_list), product_info, obj_id, foldername))

        if not work:
            active = find_active_run(_get_mongo_collection, division_idx, is_cold)
            pipeline_log(run_summary(active), phase="pipeline_run")
            return

        gpu_ids = _annotation_gpu_ids()
        gpu_slot_wait_sec = _annotation_gpu_slot_wait_sec()
        gpu_pool = get_gpu_slot_pool(gpu_ids, gpu_slot_wait_sec, logger=pipeline_log)
        slot_capacity = gpu_pool.total_slot_capacity
        max_workers = len(work)

        pipeline_log(
            f"annotation batch 시작: {len(work)}개 product, "
            f"NVML 동적 GPU{gpu_ids} 선택 + cam 순차, "
            f"GPU당 최대 {gpu_pool.max_slots_per_gpu}개/전체 최대 {slot_capacity}개 "
            f"(작업당 여유 VRAM {gpu_pool.slot_memory_fraction:.0%} 필요, "
            f"초과분 GPU 슬롯 대기, slot_timeout={gpu_slot_wait_sec}s), run_id={run_id}",
            phase="annotation",
        )
        if len(work) > slot_capacity:
            pipeline_log(
                f"annotation 큐: {len(work)}개 — {slot_capacity + 1}번째 이후는 공유 GPU 슬롯 반환 후 시작",
                phase="annotation",
            )

        def _mark_annotation_failed(foldername: str, obj_id: Optional[str], message: str) -> None:
            pipeline_log(message, phase="annotation", level="ERROR")
            if run_id and obj_id:
                update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_FAILED)

        def _annotate_one(item: tuple) -> None:
            idx, total, product_info, obj_id, foldername = item
            _bind_request_ctx(request_ctx)
            if _annotation_batch_cancelled(division_idx, is_cold, run_id):
                return

            gpu_id = None
            try:
                gpu_id = gpu_pool.acquire(foldername)
                if _annotation_batch_cancelled(division_idx, is_cold, run_id):
                    return

                if run_id and obj_id:
                    update_product_phase(
                        _get_mongo_collection, run_id, obj_id, PHASE_PENDING_ANNOTATION
                    )

                pipeline_log(
                    f"annotation 시작 ({idx}/{total}): {foldername} — GPU {gpu_id}, cam 순차",
                    phase="annotation",
                )
                ok = run_annotation_job(
                    division_idx,
                    product_info,
                    is_cold=is_cold,
                    gpu_id=gpu_id,
                    parallel_cams=False,
                )
                if _annotation_batch_cancelled(division_idx, is_cold, run_id):
                    pipeline_log(
                        f"annotation batch 중단: run_id={run_id}, last={foldername}",
                        phase="annotation",
                    )
                    return
                if not ok:
                    _mark_annotation_failed(foldername, obj_id, f"annotation 실패: {foldername}")
                    return

                if run_id and obj_id:
                    update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_AWAITING_REVIEW)
                pipeline_log(
                    f"annotation 완료 — 웹 검수 대기: {foldername} (GPU {gpu_id})",
                    phase="annotation",
                )
            except _GpuSlotTimeout as e:
                _mark_annotation_failed(foldername, obj_id, f"annotation GPU 슬롯 대기 실패: {foldername} — {e}")
            except Exception as e:
                _mark_annotation_failed(
                    foldername,
                    obj_id,
                    f"annotation 예외 발생: {foldername} — {type(e).__name__}: {e}",
                )
            finally:
                if gpu_id is not None:
                    gpu_pool.release(gpu_id, foldername)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_annotate_one, item) for item in work]
            for item, future in zip(work, futures):
                _, _, _, obj_id, foldername = item
                try:
                    future.result()
                except Exception as e:
                    _mark_annotation_failed(
                        foldername,
                        obj_id,
                        f"annotation worker 예외 미처리: {foldername} — {type(e).__name__}: {e}",
                    )

        active = find_active_run(_get_mongo_collection, division_idx, is_cold)
        pipeline_log(run_summary(active), phase="pipeline_run")
    finally:
        _release_annotation_batch(division_idx, is_cold, run_id)


def _run_skip_pipeline_for_existing_products(
    division_idx: str,
    product_list: list,
    is_cold: bool = False,
    request_ctx: Optional[RequestLogContext] = None,
    run_id: str = "",
) -> None:
    """
    isNew=0 (기학습) 상품: 어노테이션·세그먼트 생략, training_status → 4 (학습 직전).
    isNew 값은 변경하지 않음.
    """
    _bind_request_ctx(request_ctx)
    if not run_id:
        run_id = _run_id_from_ctx(request_ctx)
    if not product_list:
        return

    pipeline_log(
        f"isNew=0 상품 {len(product_list)}개 — annotation/segment skip, status → 4",
        phase="is_new",
    )

    for product_info in product_list:
        brunch_name = product_info.get("brunch_name") or f"{division_idx}_{'C' if is_cold else 'F'}"
        obj_id = product_info.get("product_object_id")
        foldername = product_info.get("foldername", "")
        current_status = str(product_info.get("training_status") or "")

        if not obj_id:
            continue

        if current_status in ("4", "5", "6", "7"):
            pipeline_log(
                f"isNew=0 skip (이미 status={current_status}): {foldername}",
                phase="is_new",
            )
            continue

        pipeline_log(
            f"isNew=0 — annotation/segment 생략, training_status → 4: {foldername} "
            f"(was status={current_status})",
            phase="is_new",
        )
        _update_training_status_in_mongo(brunch_name, obj_id, status="4")
        if run_id:
            update_product_phase(_get_mongo_collection, run_id, obj_id, PHASE_READY)

    active = find_active_run(_get_mongo_collection, division_idx, is_cold)
    pipeline_log(run_summary(active), phase="pipeline_run")
    _maybe_advance_pipeline(division_idx, is_cold, request_ctx=request_ctx, run_id=run_id)


@app.post("/v1/events/product/created", response_model=ResponseEnvelope, responses={400: {"model": ResponseEnvelope}, 404: {"model": ResponseEnvelope}, 500: {"model": ResponseEnvelope}, 503: {"model": ResponseEnvelope}})
def product_created(payload: ProductCreatedEnvelope, background_tasks: BackgroundTasks):
    division_idx = payload.DATA.division_idx
    is_cold = payload.DATA.is_cold

    request_ctx = init_request_log(
        event="product_created",
        if_sysid=payload.HEADER.IF_SYSID,
        if_id=payload.HEADER.IF_ID,
        division_idx=division_idx,
        is_cold=is_cold,
    )
    pipeline_log(f"payload: {payload.model_dump()}", phase="product_created")

    # 모든 product 조회
    try:
        all_products = _resolve_product_info_from_mongo(
            division_idx, is_cold=is_cold, raise_on_error=True
        )
    except (MongoConfigurationError, MongoQueryError) as exc:
        pipeline_log(
            f"MongoDB unavailable: {type(exc).__name__}: {exc}",
            phase="product_created",
            level="ERROR",
        )
        return make_error_response(payload.HEADER, 503, "MongoDB unavailable")

    if not all_products:
        message = f"division_idx not found: {division_idx} (is_cold={is_cold})"
        pipeline_log(message, phase="product_created", level="ERROR")
        return make_error_response(payload.HEADER, 404, message)

    pipeline_log(
        f"division_idx={division_idx}, is_cold={is_cold}, total_products={len(all_products)}",
        phase="product_created",
    )

    # training_status별로 분류
    products_status_2, products_status_3, products_status_4_plus, products_status_6 = (
        _classify_products_by_status(all_products)
    )

    pipeline_log(
        f"status 분류: status=2({len(products_status_2)}), status=3({len(products_status_3)}), "
        f"status=6({len(products_status_6)}), status>=4({len(products_status_4_plus)})",
        phase="product_created",
    )

    products_status_2_new, products_status_2_existing = _split_products_by_is_new(products_status_2)
    products_status_3_new, products_status_3_existing = _split_products_by_is_new(products_status_3)
    products_skip_to_training = products_status_2_existing + products_status_3_existing

    pipeline_log(
        f"isNew 분류: status=2 isNew=1({len(products_status_2_new)})/0({len(products_status_2_existing)}), "
        f"status=3 isNew=1({len(products_status_3_new)})/0({len(products_status_3_existing)})",
        phase="product_created",
    )

    run_id = payload.HEADER.IF_SYSID.strip()
    brunch_name = all_products[0].get("brunch_name") if all_products else f"{division_idx}_{'C' if is_cold else 'F'}"
    pipeline_run, effective_run_id, merged = merge_or_create_run_from_created(
        _get_mongo_collection,
        run_id=run_id,
        if_id=payload.HEADER.IF_ID,
        division_idx=division_idx,
        is_cold=is_cold,
        brunch_name=brunch_name,
        log_dir=request_ctx.log_dir,
        pending_annotation=products_status_2_new,
        pending_skip=products_skip_to_training,
        pending_segment=products_status_3_new,
    )
    if pipeline_run:
        run_id = effective_run_id
        if merged:
            pipeline_log(
                f"동일 division 활성 run에 요청 병합: "
                f"request_run_id={payload.HEADER.IF_SYSID}, effective_run_id={run_id}",
                phase="pipeline_run",
            )
        pipeline_log(run_summary(pipeline_run), phase="pipeline_run")
    else:
        pipeline_log("PipelineRun 생성/병합 실패 (run_id 없음)", phase="pipeline_run", level="ERROR")

    # isNew=0: 어노테이션·세그먼트 없이 학습 직전(status=4)으로
    if products_skip_to_training:
        _add_background_task(
            background_tasks,
            _run_skip_pipeline_for_existing_products,
            division_idx,
            products_skip_to_training,
            is_cold,
            request_ctx,
            run_id,
        )

    # isNew=1 + status=2: 자동 어노테이션 (웹 검수 후 segment)
    if products_status_2_new:
        pipeline_log(
            f"isNew=1 + status=2 → annotation {len(products_status_2_new)}개 시작",
            phase="product_created",
        )
        _schedule_annotation_batch(
            background_tasks,
            division_idx,
            products_status_2_new,
            is_cold,
            request_ctx,
            run_id,
        )

    # isNew=1 + status=3: segment (웹 검수 완료 후)
    if products_status_3_new:
        pipeline_log(
            f"isNew=1 + status=3 → segment {len(products_status_3_new)}개 시작",
            phase="product_created",
        )
        for product_info in products_status_3_new:
            _add_background_task(
                background_tasks,
                _run_segment_job,
                division_idx,
                product_info,
                is_cold,
                request_ctx,
                run_id,
            )

    # 이미 segment까지 끝난 상태로 들어온 경우 (예: 전원 status>=4) 바로 advance
    if (
        not products_status_2_new
        and not products_status_3_new
        and not products_skip_to_training
    ):
        _add_background_task(
            background_tasks, _maybe_advance_pipeline, division_idx, is_cold, request_ctx, run_id
        )

    return make_response(payload.HEADER, "S", "accepted")

# -----------------------------
# B) Web -> AI : 어노테이션 완료 콜백
# 최소: division_idx, product_idx is_cold
# -----------------------------
class AnnotationCompletedData(BaseModel):
    division_idx: str
    product_idx: str
    train_product_idx: Optional[str] = None
    training_idx: Optional[str] = None
    trainProductIdx: Optional[str] = None
    TrainingIdx: Optional[str] = None
    is_cold: bool

class AnnotationCompletedEnvelope(BaseModel):
    HEADER: Header
    DATA: AnnotationCompletedData

@app.post("/v1/events/annotation/completed", response_model=ResponseEnvelope, responses={400: {"model": ResponseEnvelope}, 404: {"model": ResponseEnvelope}, 409: {"model": ResponseEnvelope}, 500: {"model": ResponseEnvelope}, 503: {"model": ResponseEnvelope}})
def annotation_completed(payload: AnnotationCompletedEnvelope, background_tasks: BackgroundTasks):
    division_idx = payload.DATA.division_idx
    product_idx = payload.DATA.product_idx
    train_product_idx = (
        payload.DATA.train_product_idx
        or payload.DATA.training_idx
        or payload.DATA.trainProductIdx
        or payload.DATA.TrainingIdx
    )
    train_product_idx = str(train_product_idx).strip() if train_product_idx is not None else ""
    is_cold = payload.DATA.is_cold

    active_run = find_active_run(_get_mongo_collection, division_idx, is_cold)
    if active_run and active_run.get("log_dir"):
        request_ctx = resume_request_log(
            active_run,
            event="annotation_completed",
            if_sysid=payload.HEADER.IF_SYSID,
            if_id=payload.HEADER.IF_ID,
            extra={
                "product_idx": product_idx,
                "train_product_idx": train_product_idx,
                "linked_run_id": active_run.get("run_id"),
            },
        )
        run_id = str(active_run.get("run_id") or "")
    else:
        run_id = ""
        request_ctx = init_request_log(
            event="annotation_completed",
            if_sysid=payload.HEADER.IF_SYSID,
            if_id=payload.HEADER.IF_ID,
            division_idx=division_idx,
            is_cold=is_cold,
            extra={"product_idx": product_idx, "train_product_idx": train_product_idx},
        )
        pipeline_log("active PipelineRun 없음 — 독립 로그로 처리", phase="pipeline_run", level="ERROR")
    pipeline_log(f"payload: {payload.model_dump()}", phase="annotation_completed")

    # 필수 필드 검증
    if not division_idx or not product_idx:
        pipeline_log("division_idx or product_idx is missing", phase="annotation_completed", level="ERROR")
        return make_error_response(payload.HEADER, 400, "division_idx or product_idx is missing")

    pipeline_log(
        f"웹 어노테이션 검수 완료: division_idx={division_idx}, "
        f"product_idx={product_idx}, train_product_idx={train_product_idx or '-'}",
        phase="annotation_completed",
    )

    # product_idx + trainProductIdx로 product_object_id 찾기 (프로모션 중복 productIdx 대응)
    from bson import ObjectId
    products_collection_name = os.getenv("MONGO_PRODUCTSLIST_COLLECTION", "ProductsList")
    products_col = _get_mongo_collection(products_collection_name)
    
    if products_col is None:
        return make_error_response(payload.HEADER, 503, "MongoDB connection failed")

    product_query = {"productIdx": product_idx}
    if train_product_idx:
        train_idx_values = [train_product_idx]
        try:
            train_idx_values.append(int(train_product_idx))
        except (TypeError, ValueError):
            pass
        product_query["trainProductIdx"] = {"$in": train_idx_values}

    product_docs = list(products_col.find(product_query))
    if not train_product_idx and len(product_docs) > 1:
        pipeline_log(
            f"product_idx 중복으로 완료 대상 특정 불가: product_idx={product_idx}, "
            f"candidates={len(product_docs)}",
            phase="annotation_completed",
            level="ERROR",
        )
        return make_error_response(
            payload.HEADER,
            409,
            f"ambiguous product_idx; train_product_idx required: {product_idx}",
        )

    product_doc = product_docs[0] if product_docs else None
    if not product_doc:
        if train_product_idx:
            return make_error_response(
                payload.HEADER,
                404,
                f"product_idx/train_product_idx not found: {product_idx}/{train_product_idx}",
            )
        return make_error_response(payload.HEADER, 404, f"product_idx not found: {product_idx}")

    product_object_id = str(product_doc.get("_id"))
    pipeline_log(f"product_object_id={product_object_id}", phase="annotation_completed")

    # 실제 DeviceTypeList 문서에서 brunchName 조회 (suffix 조합 의존 제거)
    device_collection_name = os.getenv("MONGO_DEVICETYPELIST_COLLECTION", "DeviceTypeList")
    device_col = _get_mongo_collection(device_collection_name)
    brunch_name = None
    if device_col is not None:
        device_doc = device_col.find_one(
            {"divisionIdx": division_idx, "products.product": ObjectId(product_object_id)},
            sort=[("trainingDate", -1), ("_id", -1)],
        )
        if device_doc:
            brunch_name = device_doc.get("brunchName")
    if not brunch_name:
        brunch_name = f"{division_idx}_{'C' if is_cold else 'F'}"
        pipeline_log(f"brunchName 조회 실패, fallback 사용: {brunch_name}", phase="annotation_completed")

    # training_status를 3으로 업데이트 + IF_07
    success = _update_training_status_in_mongo(brunch_name, product_object_id, status="3")
    if not success:
        pipeline_log("training_status update failed", phase="annotation_completed", level="ERROR")
        return make_error_response(payload.HEADER, 500, "training_status update failed")

    if run_id:
        update_product_phase(_get_mongo_collection, run_id, product_object_id, PHASE_SEGMENTING)

    pipeline_log(
        f"training_status → 3, segment 시작: brunch_name={brunch_name}, product_idx={product_idx}",
        phase="annotation_completed",
    )

    product_info = _build_product_info(product_doc, brunch_name, training_status="3")
    background_tasks.add_task(
        _run_segment_job,
        division_idx,
        product_info,
        is_cold,
        request_ctx,
        run_id,
    )

    return make_response(payload.HEADER, "S", "accepted")


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    for path in (
        "/v1/events/product/created",
        "/v1/events/annotation/completed",
    ):
        post = schema.get("paths", {}).get(path, {}).get("post", {})
        post.get("responses", {}).pop("422", None)
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2140, reload=False)
