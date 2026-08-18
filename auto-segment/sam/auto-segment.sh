#!/bin/bash

# GPU 지정 (예: GPU 0만 사용)
export CUDA_VISIBLE_DEVICES="${PIPELINE_CUDA_VISIBLE_DEVICES:-1}"
export TORCHDYNAMO_DISABLE=1

# ===== 설정 =====
# segment_job.py에서 환경변수로 전달됨
VENV_PYTHON="${VENV_PYTHON:-python}"  # 실행할 python (컨테이너는 compose에서 python 전달)
SEGMENT_BASE_DIR="${SEGMENT_BASE_DIR:-/app/segment_work}"
CLASS_NAME="${CLASS_NAME:-}"  # foldername (예: 1_BOTTLE_KWANGDONG_SAMDASOO_500ML_20260129_175426)
IS_COLD="${IS_COLD:-FALSE}"  # TRUE or FALSE
IMAGES_DIR="${IMAGES_DIR:-}"  # 이미지 디렉토리 (segment_job.py에서 다운로드)
LABELS_DIR="${LABELS_DIR:-}"  # 라벨 디렉토리 (segment_job.py에서 다운로드)
TRAIN_PRODUCT_IDX="${TRAIN_PRODUCT_IDX:-}"  # train_product_idx (예: 1)
PRODUCT_ENG_NAME="${PRODUCT_ENG_NAME:-}"  # product_eng_name (예: BOTTLE_KWANGDONG_SAMDASOO_500ML)
DIVISION_IDX="${DIVISION_IDX:-}"  # division_idx (예: DI17246510607993400)

# ===== MinIO 설정 =====
# MinIO 서버 정보 (환경변수 또는 직접 설정)
MINIO_ENDPOINT="${MINIO_ENDPOINT:-}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-}"
MINIO_BUCKET="${MINIO_BUCKET:-chaiimage}"
MINIO_BASE_PREFIX="${MINIO_BASE_PREFIX:-productSeg}"  # segment 결과 업로드 경로
MINIO_SECURE="${MINIO_SECURE:-false}"

# ===== MongoDB 설정 (필요시) =====
MONGO_URI="${MONGO_URI:-}"
MONGO_DB="${MONGO_DB:-CHAI}"

# ===== 검증 =====
if [ -z "$CLASS_NAME" ]; then
    echo "ERROR: CLASS_NAME 환경변수가 설정되지 않았습니다"
    exit 1
fi

if [ -z "$TRAIN_PRODUCT_IDX" ]; then
    echo "ERROR: TRAIN_PRODUCT_IDX 환경변수가 설정되지 않았습니다"
    exit 1
fi

if [ -z "$PRODUCT_ENG_NAME" ]; then
    echo "ERROR: PRODUCT_ENG_NAME 환경변수가 설정되지 않았습니다"
    exit 1
fi

if [ -z "$DIVISION_IDX" ]; then
    echo "ERROR: DIVISION_IDX 환경변수가 설정되지 않았습니다"
    exit 1
fi

if [ -z "$IMAGES_DIR" ] || [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: IMAGES_DIR이 설정되지 않았거나 디렉토리가 없습니다: ${IMAGES_DIR}"
    exit 1
fi

if [ -z "$LABELS_DIR" ] || [ ! -d "$LABELS_DIR" ]; then
    echo "ERROR: LABELS_DIR이 설정되지 않았거나 디렉토리가 없습니다: ${LABELS_DIR}"
    exit 1
fi

# 이미지 파일 개수 확인 (하위 디렉토리 포함)
IMG_COUNT=$(find "$IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
if [ "$IMG_COUNT" -eq 0 ]; then
    echo "ERROR: 이미지 파일이 없습니다: ${IMAGES_DIR}"
    exit 1
fi

# 라벨 파일 개수 확인 (하위 디렉토리 포함)
LBL_COUNT=$(find "$LABELS_DIR" -type f -name "*.txt" 2>/dev/null | wc -l)
if [ "$LBL_COUNT" -eq 0 ]; then
    echo "WARN: 라벨 파일이 없습니다: ${LABELS_DIR}"
fi

echo "=========================================="
echo "Segment 합성 작업 시작"
echo "=========================================="
echo "CLASS_NAME: ${CLASS_NAME}"
echo "TRAIN_PRODUCT_IDX: ${TRAIN_PRODUCT_IDX}"
echo "PRODUCT_ENG_NAME: ${PRODUCT_ENG_NAME}"
echo "DIVISION_IDX: ${DIVISION_IDX}"
echo "IS_COLD: ${IS_COLD}"
echo "IMAGES_DIR: ${IMAGES_DIR} (${IMG_COUNT}개 이미지)"
echo "LABELS_DIR: ${LABELS_DIR} (${LBL_COUNT}개 라벨)"
echo "SEGMENT_BASE_DIR: ${SEGMENT_BASE_DIR}"
echo "=========================================="

# ===== 출력 디렉토리 설정 =====
OUT_BASE="${SEGMENT_BASE_DIR}/${CLASS_NAME}/output"
OUT_SEGMENT_DIR="${OUT_BASE}/segments"
OUT_MASK_DIR="${OUT_BASE}/masks"
OUT_VISUALIZED_DIR="${OUT_BASE}/visualized"

mkdir -p "${OUT_SEGMENT_DIR}"
mkdir -p "${OUT_MASK_DIR}"
mkdir -p "${OUT_VISUALIZED_DIR}"

# ===== Segment 합성 작업 실행 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEGMENT_PY="${SCRIPT_DIR}/segment_synthesis.py"

if [ ! -f "$SEGMENT_PY" ]; then
    echo "ERROR: segment_synthesis.py를 찾을 수 없습니다: ${SEGMENT_PY}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Segment 합성 작업 실행"
echo "=========================================="
echo "Python 스크립트: ${SEGMENT_PY}"
echo "=========================================="

"$VENV_PYTHON" "$SEGMENT_PY" \
    --img_dir "${IMAGES_DIR}" \
    --lbl_dir "${LABELS_DIR}" \
    --out_segment_dir "${OUT_SEGMENT_DIR}" \
    --out_mask_dir "${OUT_MASK_DIR}" \
    --out_visualized_dir "${OUT_VISUALIZED_DIR}" \
    --train_product_idx "${TRAIN_PRODUCT_IDX}" \
    --product_eng_name "${PRODUCT_ENG_NAME}" \
    --division_idx "${DIVISION_IDX}" \
    --is_cold "${IS_COLD}" \
    --skip_minio_upload \
    --use_side

PY_EXIT=$?
if [ "$PY_EXIT" -ne 0 ]; then
    echo "ERROR: Segment 합성 작업 실패 (종료 코드: ${PY_EXIT})"
    exit 1
fi

echo ""
echo "=========================================="
echo "Segment 합성 작업 완료"
echo "=========================================="

# ===== MinIO 업로드 (결과가 있을 경우) =====
upload_segment_pkl_to_minio() {
    local endpoint="$1"
    local access_key="$2"
    local secret_key="$3"
    local bucket="$4"
    local secure="$5"
    local base_prefix="$6"
    local train_product_idx="$7"
    local product_eng_name="$8"
    local pkl_file="$9"

    "$VENV_PYTHON" - "$endpoint" "$access_key" "$secret_key" "$bucket" "$secure" "$base_prefix" "$train_product_idx" "$product_eng_name" "$pkl_file" <<'PY'
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from minio import Minio
except Exception as e:
    print(f"[error] minio import failed: {e}", file=sys.stderr)
    sys.exit(2)

endpoint, access_key, secret_key, bucket, secure_s, base_prefix, train_product_idx, product_eng_name, pkl_file = sys.argv[1:10]
secure = str(secure_s).lower() == "true"

if not os.path.isfile(pkl_file):
    print(f"[error] PKL 파일이 없습니다: {pkl_file}", file=sys.stderr)
    sys.exit(2)

# 날짜 형식: YYYYMMDD
date_str = datetime.now().strftime("%Y%m%d")
# 파일명: train_product_idx_engname_날짜.pkl
pkl_filename = f"{train_product_idx}_{product_eng_name}_{date_str}.pkl"

client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
if not client.bucket_exists(bucket):
    client.make_bucket(bucket)

prefix = base_prefix.strip("/")
object_name = f"{prefix}/{pkl_filename}" if prefix else pkl_filename

client.fput_object(bucket, object_name, pkl_file)
print(f"[info] MinIO 업로드 완료: {bucket}/{object_name}")
PY
}

# PKL 파일 찾기 (segment_synthesis.py가 생성한 파일)
# 최종 통합 PKL 파일: {train_product_idx}_{product_eng_name}_{date}.pkl
PKL_PATTERN="${TRAIN_PRODUCT_IDX}_${PRODUCT_ENG_NAME}_*.pkl"
PKL_FOUND=$(find "${OUT_SEGMENT_DIR}" -maxdepth 1 -name "${PKL_PATTERN}" -type f | head -1)

if [ -n "$MINIO_ENDPOINT" ] && [ -n "$MINIO_ACCESS_KEY" ] && [ -n "$MINIO_SECRET_KEY" ]; then
    MINIO_ENDPOINT=$(echo "$MINIO_ENDPOINT" | tr -d ' ')
    echo ""
    echo "=========================================="
    echo "MinIO 업로드 시작"
    echo "=========================================="
    
    if [ -n "$PKL_FOUND" ] && [ -f "$PKL_FOUND" ]; then
        echo "[info] PKL 파일 발견: ${PKL_FOUND}"
        upload_segment_pkl_to_minio \
            "${MINIO_ENDPOINT}" \
            "${MINIO_ACCESS_KEY}" \
            "${MINIO_SECRET_KEY}" \
            "${MINIO_BUCKET}" \
            "${MINIO_SECURE}" \
            "${MINIO_BASE_PREFIX}" \
            "${TRAIN_PRODUCT_IDX}" \
            "${PRODUCT_ENG_NAME}" \
            "${PKL_FOUND}"
        
        UPLOAD_EXIT=$?
        if [ "$UPLOAD_EXIT" -ne 0 ]; then
            echo "[warn] MinIO 업로드 실패"
        else
            echo "[info] MinIO 업로드 완료"
        fi
    else
        echo "[warn] PKL 파일을 찾을 수 없습니다: ${OUT_SEGMENT_DIR}"
    fi
fi

echo ""
echo "=========================================="
echo "전체 작업 완료!"
echo "=========================================="
