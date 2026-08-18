FROM python:3.14-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PNT_API_HOST=0.0.0.0 \
    PNT_API_PORT=2140 \
    PNT_LOG_DIR=/app/logs \
    PNT_LOG_FILE=api.log \
    PNT_LOG_TO_FILE=false \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        curl \
        git \
        procps \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.txt

COPY main.py ./
COPY gpu_scheduler.py ./
COPY http_logging.py ./
COPY pipeline_run.py annotation_batch_queue.py request_log.py job_log.py ./
COPY annotation_job.py review_package_sync.py segment_job.py synthesis_job.py training_job.py class_schema.py deploy_job.py ./
COPY pnt_login.py ./
COPY auto-labeling/sam3_with_dino ./auto-labeling/sam3_with_dino
COPY auto-labeling/background ./auto-labeling/background
COPY auto-segment/sam ./auto-segment/sam

RUN mkdir -p logs img_download runs/sam3_with_dino segment_work \
        /cache/huggingface /cache/torch /cache/ultralytics \
    && chmod +x auto-labeling/sam3_with_dino/*.py \
    && chmod +x auto-labeling/background/*.py \
    && chmod +x auto-segment/sam/*.sh \
    && python -c "import cv2, fastapi, minio, pymongo, pynvml, sam3, segment_anything, torch, transformers, ultralytics"

EXPOSE 2140

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PNT_API_PORT}/health" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2140", "--no-access-log"]
