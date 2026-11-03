# CRK AI PNT 파이프라인

이 저장소는 상품 생성 이벤트를 받아 자동 어노테이션, 웹 검수, 상품 세그먼트 생성, YOLO 학습·검증, 모델 배포까지 수행하는 FastAPI 기반 AI 파이프라인입니다.

현재 자동 어노테이션은 **Grounding DINO + SAM3** 조합을 사용합니다. Docker Compose가 기본 실행 방식이며 API는 호스트의 `2140` 포트에서 동작합니다.

## 1. 현재 운영 구성

| 항목 | 구성 |
| --- | --- |
| API | FastAPI + Uvicorn |
| Python | 3.14 |
| 자동 어노테이션 | Grounding DINO + SAM3 Video Tracking |
| 학습 | Ultralytics YOLO |
| 데이터 저장 | MongoDB, MinIO |
| 로그 수집 | Docker json-file -> Alloy -> Loki -> Grafana |
| GPU | Docker에서 전체 GPU 노출, 어노테이션 GPU는 환경변수로 제어 |
| API 포트 | `2140` |
| Loki 포트 | `127.0.0.1:3100` |

실행되는 컨테이너는 다음과 같습니다.

- `crk-0-pnt-api`: FastAPI 및 AI 파이프라인
- `pnt-loki`: 로그 저장 및 LogQL 조회
- `pnt-alloy`: Docker API 로그 수집 및 Loki 전송

## 2. 전체 구조

```text
FastAPI 상품 생성 이벤트
  -> MongoDB에서 division 및 상품 조회
  -> 상품별 training_status / isNew 분류
  -> Grounding DINO로 손과 상품 후보 탐지
  -> SAM3로 후보 마스크 생성 및 bbox 보정
  -> YOLO Detection / Segmentation 라벨 생성
  -> MinIO에 이미지와 어노테이션 업로드
  -> 웹 검수 대기
  -> 어노테이션 완료 콜백
  -> 상품 세그먼트 생성
  -> 합성 데이터 생성
  -> YOLO 학습 및 검증
  -> 검증 통과 모델 MinIO 업로드
  -> PNT 배포 완료 이벤트 전송
```

로그 흐름은 다음과 같습니다.

```text
Uvicorn stdout/stderr
  -> Docker json-file 로그
  -> Grafana Alloy
  -> Loki
  -> Grafana 대시보드
```

## 3. 디렉터리와 주요 파일

```text
pnt-api/
├── main.py                         # FastAPI 엔드포인트와 전체 orchestration
├── annotation_job.py               # SAM3+DINO 어노테이션 작업 및 MinIO 연동
├── gpu_scheduler.py               # NVML 기반 GPU 상태 조회·동적 할당
├── segment_job.py                  # 상품 세그먼트 생성
├── synthesis_job.py                # 학습용 합성 데이터 생성
├── training_job.py                 # YOLO 학습 및 검증
├── deploy_job.py                   # 모델 MinIO 업로드 및 배포 처리
├── pipeline_run.py                 # 요청별 파이프라인 상태 관리
├── request_log.py                  # 요청 로그 컨텍스트
├── job_log.py                      # 작업 로그
├── pnt_login.py                    # PNT 인증 처리
├── auto-labeling/
│   └── sam3_with_dino/
│       ├── asis_pipeline.py        # 운영 2-pass 자동 어노테이션 실행기
│       ├── run_product.py          # 상품/카메라 단위 내부 실행기
│       ├── video_annotate.py       # DINO seed + SAM3 양방향 Video Tracking
│       ├── auto_annotate.py        # 이미지 단위 fallback 및 후보 선택
│       ├── export_web_annotations.py # COCO 웹 업로드 JSON 변환
│       └── benchmark.py            # 상품 유형별 개발/홀드아웃 품질 평가
├── auto-segment/
│   └── sam/                        # 검수 완료 상품 세그먼트 생성 코드
├── img_download/                   # 어노테이션 입력 이미지
├── runs/                           # SAM3/DINO 및 YOLO 결과
├── segment_work/                   # 세그먼트 임시 작업 공간
├── logs/
│   ├── requests/                   # 요청별 pipeline.log
│   └── pipeline_state/             # 요청별 파이프라인 상태 JSON
├── observability/
│   ├── alloy-config.alloy          # Docker 로그 수집 설정
│   ├── loki-config.yaml            # Loki 저장·보존 설정
│   └── grafana-dashboard.json      # Grafana 대시보드 원본
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                # Docker Python 전체 의존성 lock
└── .env                            # 인증정보와 외부 시스템 설정, Git에 커밋 금지
```

`logs`, `runs`, `segment_work`, `observability`, `img_download`는 Docker bind mount 또는 런타임 데이터 경로이므로 삭제하면 안 됩니다.

## 4. 실행 전 준비

### 필수 조건

- Docker Engine
- Docker Compose 플러그인
- NVIDIA Container Toolkit
- 호스트 NVIDIA 드라이버
- MongoDB 및 MinIO 접근 권한
- gated 모델 사용이 가능한 Hugging Face 토큰

확인 명령:

```bash
docker version
docker compose version
nvidia-smi
docker info | grep -i runtime
```

### 환경변수

루트의 `.env`를 사용합니다. 실제 비밀값은 README나 Git에 기록하지 않습니다.

주요 항목:

| 변수 | 용도 |
| --- | --- |
| `MONGO_URI` | MongoDB 접속 URI |
| `MONGO_DB` | MongoDB 데이터베이스 |
| `MONGO_PRODUCTSLIST_COLLECTION` | 상품 컬렉션, 기본값 `ProductsList` |
| `MONGO_DEVICETYPELIST_COLLECTION` | 장비/학습 상태 컬렉션, 기본값 `DeviceTypeList` |
| `MINIO_ENDPOINT` | MinIO 주소 |
| `MINIO_ACCESS_KEY` | MinIO access key |
| `MINIO_SECRET_KEY` | MinIO secret key |
| `MINIO_BUCKET` | 이미지·모델 버킷, 기본값 `chaiimage` |
| `MINIO_SECURE` | HTTPS 사용 여부 |
| `PNT_USER_ID`, `PNT_USER_PASSWORD` | PNT 로그인 정보 |
| `PNT_ACCESS_TOKEN` | 고정 토큰을 사용할 경우 설정 |
| `HF_TOKEN` | Hugging Face 모델 접근 토큰 |
| `HTTP_LOG_BODY_MAX_BYTES` | 요청·응답 payload 로그 최대 크기, 기본값 `65536` bytes |
| `HTTP_LOG_HEALTH` | `/health` 구조화 로그 포함 여부, 기본값 `false` |
| `HTTP_LOG_EXCLUDED_PATHS` | payload 로그 제외 경로. 기본값 `/health,/docs,/openapi.json,/redoc` |
| `PIPELINE_ANNOTATION_GPUS` | 어노테이션 작업에 사용할 GPU 목록 |
| `ANNOTATION_GPU_SLOT_MEMORY_FRACTION` | 작업 1개가 요구하는 GPU 여유 VRAM 비율, 기본값 `0.50` |
| `ANNOTATION_GPU_MAX_SLOTS_PER_DEVICE` | GPU 하나에 예약할 수 있는 최대 작업 수, 기본값 `2` |
| `ANNOTATION_LOCK_WAIT_SEC` | 직접 실행·복구 경로의 division batch lock 대기 시간. 일반 product/created 재요청은 직렬 큐를 사용 |
| `ANNOTATION_GPU_SLOT_WAIT_SEC` | GPU 슬롯 대기 시간, 기본값 `43200`초(12시간) |
| `ANNOTATION_GPU_MIN_FREE_MEMORY_MIB` | GPU 할당에 필요한 최소 여유 VRAM(MiB), 기본값 `24576` |
| `ANNOTATION_MINIO_UPLOAD_WORKERS` | 웹 검수 원본 이미지 MinIO 병렬 업로드 수, 기본값 `4` |
| `ANNOTATION_GPU_MAX_UTILIZATION` | GPU 할당이 가능한 최대 사용률(%), 기본값 `80` |
| `ANNOTATION_GPU_POLL_INTERVAL_SEC` | 사용 가능한 GPU 재조회 주기(초), 기본값 `5` |
| `ANNOTATION_GPU_STATUS_LOG_INTERVAL_SEC` | GPU 대기 상태 로그 주기(초), 기본값 `60` |

SAM3+DINO 공통 기준은 `docker-compose.yml`에서 설정합니다.

| 변수 | 현재 기본값 | 의미 |
| --- | ---: | --- |
| `SAM3_DINO_MODEL` | `IDEA-Research/grounding-dino-base` | Grounding DINO 모델 |
| `SAM3_DINO_RUNNER` | `asis_pipeline.py` | 운영 2-pass 자동 어노테이션 실행기 |
| `SAM3_DINO_BACKEND` | `video` | 운영 경로는 SAM3 Video Tracking 사용 |
| `SAM3_DINO_ASIS_FILL_THRESHOLD` | `0.45` | pass 1의 SAM3 mask/DINO hand box 최소 채움 비율 |
| `SAM3_DINO_ASIS_EDGE_AREA_RATIO` | `0.30` | 화면 가장자리에 닿은 큰 손 bbox 제거 면적 기준 |
| `SAM3_DINO_ASIS_SHORT_RUN` | `3` | 제거할 짧은 손 탐지 연속 구간의 최대 프레임 수 |
| `SAM3_DINO_ASIS_DINO_BATCH` | `4` | 프레임별 DINO 배치 추론 크기 |
| `SAM3_DINO_HAND_BOX_THRESHOLD` | `0.25` | 손 후보 탐지 임계값 |
| `SAM3_DINO_OBJECT_BOX_THRESHOLD` | `0.22` | 상품 후보 탐지 임계값 |
| `SAM3_DINO_TEXT_THRESHOLD` | `0.20` | 텍스트 정합 임계값 |
| `SAM3_DINO_CONFIDENCE` | `0.35` | SAM3 마스크 신뢰도 |
| `SAM3_DINO_HAND_PROMPTS` | `human hand|left hand|right hand|palm|fingers` | DINO 손 후보 프롬프트 목록 |
| `SAM3_DINO_SAM3_HAND_PROMPT` | `human hand and fingers` | SAM3 Video Tracking 손 프롬프트 |
| `SAM3_DINO_AREA_WEIGHT` | `0.15` | SAM3 mask와 prompt bbox 면적 유사도 가중치 |
| `SAM3_DINO_MIN_PROMPT_IOU` | `0.15` | SAM3 mask bbox 최소 prompt IoU |
| `SAM3_DINO_MIN_AREA_RATIO` | `0.15` | SAM3 mask/DINO bbox 최소 면적비 |
| `SAM3_DINO_MAX_AREA_RATIO` | `4.0` | SAM3 mask/DINO bbox 최대 면적비 |
| `SAM3_DINO_TEMPORAL_OBJECT_IOU_WEIGHT` | `0.20` | 이전 프레임 상품 bbox 연속성 가중치 |
| `SAM3_DINO_TEMPORAL_DECAY` | `0.75` | 탐지 누락 프레임마다 적용하는 시간 가중치 감쇠 |
| `SAM3_DINO_TEMPORAL_MAX_GAP` | `5` | 이전 상품 bbox를 유지하는 최대 프레임 수 |
| `SAM3_DINO_MIN_TEMPORAL_COMPATIBILITY` | `0.30` | 낮은 점수 상품을 유지할 최소 시간 호환성 |
| `SAM3_DINO_MIN_HAND_OBJECT_RELATION` | `0.08` | 낮은 점수 상품을 유지할 최소 손-상품 관계 |
| `SAM3_DINO_OBJECT_SOLO_THRESHOLD` | `0.38` | 손·시간 증거가 없는 상품의 최소 점수 |
| `SAM3_DINO_HAND_SOLO_THRESHOLD` | `0.40` | 상품 증거가 없는 손의 최소 점수 |
| `SAM3_DINO_VIDEO_PRODUCT_SEED_THRESHOLD` | `0.24` | 상품 Video Tracking seed 최소 점수 |
| `SAM3_DINO_VIDEO_HAND_SEED_THRESHOLD` | `0.30` | 손 Video Tracking seed 최소 점수 |
| `SAM3_DINO_VIDEO_MIN_OUTPUT_SCORE` | `0.40` | Video Tracking 출력 최소 점수 |
| `SAM3_DINO_VIDEO_SEED_SUPPORT_RADIUS` | `3` | seed 주변의 DINO 시간축 지지 범위 |
| `SAM3_DINO_CROSS_CLASS_DUPLICATE_IOU` | `0.88` | 손·상품 bbox 중복 제거 IoU 기준 |
| `SAM3_DINO_CROSS_CLASS_DUPLICATE_AREA_SIMILARITY` | `0.85` | 중복 제거 면적 유사도 기준 |
| `SAM3_DINO_CROSS_CLASS_DUPLICATE_MASK_IOU` | `0.80` | 손·상품 mask 중복 제거 IoU 기준 |
| `SAM3_DINO_BACKGROUND_REFERENCE_ROOT` | `/app/img_download/_backgrounds/{division_idx}` | division·카메라별 빈 배경 참조 루트 |
| `SAM3_DINO_BACKGROUND_PIXEL_THRESHOLD` | `22` | 배경과 현재 프레임의 픽셀 차이를 전경으로 보는 기준 |
| `SAM3_DINO_BACKGROUND_MIN_HAND_FOREGROUND` | `0.06` | 손 후보·mask가 포함해야 하는 최소 전경 비율 |
| `SAM3_DINO_BACKGROUND_MIN_OBJECT_FOREGROUND` | `0.08` | 상품 후보·mask가 포함해야 하는 최소 전경 비율 |
| `SAM3_DINO_COMPACT_HAND_MAX_PRODUCT_GAP` | `0.18` | 손 세부 후보와 상품 사이의 최대 정규화 간격 |
| `SAM3_DINO_COMPACT_HAND_EXPANSION` | `0.30` | 손가락·손바닥 후보를 손 bbox로 확장하는 비율 |
| `SAM3_DINO_HAND_MASK_SUPPORT_EXPANSION` | `0.15` | 전체 손 seed의 기본 mask 허용 확장 비율 |
| `SAM3_DINO_HAND_PARTIAL_MASK_SUPPORT_EXPANSION` | `0.45` | 손가락·손바닥 seed의 최소 동적 확장 비율 |
| `SAM3_DINO_HAND_MAX_MASK_SUPPORT_EXPANSION` | `0.60` | 손 mask 동적 확장 상한 |
| `SAM3_DINO_HAND_COMPONENT_MIN_AREA_RATIO` | `0.04` | 보조 손 컴포넌트의 최소 상대 면적 |
| `SAM3_DINO_HAND_COMPONENT_MAX_GAP` | `0.08` | 분리된 손 컴포넌트를 같은 손으로 묶는 최대 간격 |
| `SAM3_DINO_HAND_COMPONENT_PRODUCT_GAP` | `0.05` | 상품에 가려진 손 컴포넌트를 유지하는 상품 인접 기준 |
| `SAM3_DINO_HAND_COLLAPSE_AREA_RATIO` | `0.45` | 최근 정상 손 또는 DINO 전체 손 대비 붕괴 면적 기준 |
| `SAM3_DINO_HAND_COLLAPSE_DIMENSION_RATIO` | `0.50` | 손 폭·높이 붕괴 기준 |
| `SAM3_DINO_HAND_COLLAPSE_HISTORY_MIN` | `4` | 시간축 기준을 만들 최소 정상 프레임 수 |
| `SAM3_DINO_HAND_COLLAPSE_HISTORY_WINDOW` | `9` | 추적 붕괴 판정에 사용할 최근 정상 프레임 수 |
| `SAM3_DINO_HAND_RECOVERY_MIN_AREA_RATIO` | `0.65` | 재시드 후보의 로컬 정상 손 대비 최소 면적비 |
| `SAM3_DINO_HAND_RECOVERY_SEARCH_RADIUS` | `60` | 붕괴 구간 앞뒤 DINO 전체 손 seed 탐색 범위 |
| `SAM3_DINO_HAND_RECOVERY_CONTEXT_FRAMES` | `45` | 재시드 후 다시 추적할 붕괴 구간 주변 범위 |
| `SAM3_DINO_HAND_RECOVERY_MAX_ANCHORS` | `0` | 추가 SAM3 재시드 상한. 기본 비활성화이며 검증된 실험에서만 1 이상 사용 |
| `SAM3_DINO_HAND_RECOVERY_MERGE_GAP` | `5` | 인접 붕괴 프레임을 한 복구 구간으로 묶는 간격 |

전체 손 DINO 후보를 seed로 우선하고, 손가락·손바닥 후보는 해당 bbox를 보완하는 근거로만 사용합니다. 상품에 의해 손 mask가 분리된 경우 유효 컴포넌트를 유지하며 Detection bbox는 그 합집합으로 생성합니다. SAM3 추적 결과의 자동 재시드는 팔·상체로 mask가 번지는 과대 박스를 만들 수 있어 기본 비활성화합니다. 별도 pilot에서 검증할 때만 `SAM3_DINO_HAND_RECOVERY_MAX_ANCHORS`를 1 이상으로 설정합니다.

## 5. Docker 실행

### 구성 검증

```bash
cd /data/aipnt/pnt-api
docker compose config --quiet
```

### 빌드 및 실행

```bash
docker compose up -d --build
```

코드나 requirements 변경 후 API만 재빌드할 때:

```bash
docker compose build pnt-api
docker compose up -d --force-recreate --no-build pnt-api
```

### 상태 확인

```bash
docker compose ps
curl -fsS http://127.0.0.1:2140/health
```

정상 응답:

```json
{"ok": true}
```

API 문서:

- Swagger UI: `http://<SERVER_IP>:2140/docs`
- OpenAPI JSON: `http://<SERVER_IP>:2140/openapi.json`

### 중지 및 재시작

```bash
docker compose stop
docker compose start
```

```bash
docker compose restart pnt-api
```

`docker compose down`은 컨테이너와 네트워크를 제거합니다. 모델 캐시와 Loki 데이터를 유지해야 하므로 일반 재시작에는 `restart`를 사용합니다. `down -v`는 named volume 데이터까지 삭제하므로 사용하지 않습니다.

## 6. API 명세

### `GET /health`

프로세스 상태를 확인합니다.

### `POST /v1/events/product/created`

상품 생성 이벤트를 받아 division 전체 파이프라인을 시작합니다.

```json
{
  "HEADER": {
    "IF_ID": "IF_EDGE_01",
    "IF_SYSID": "EDGEPC-20260719-000001",
    "IF_HOST": "EDGEPC",
    "IF_DATE": "20260719000001"
  },
  "DATA": {
    "division_idx": "DI17798460900133031",
    "is_cold": true
  }
}
```

정상 접수 응답:

```json
{
  "HEADER": {
    "IF_ID": "IF_EDGE_01",
    "IF_SYSID": "EDGEPC-20260719-000001",
    "IF_HOST": "EDGEPC",
    "IF_DATE": "20260719000001"
  },
  "DATA": {
    "result_cd": "S",
    "result_msg": "accepted"
  }
}
```

처리 흐름:

1. MongoDB에서 division과 모든 상품을 조회합니다.
2. `training_status`와 `isNew` 기준으로 상품을 분류합니다.
3. 동일 division에 활성 run이 있으면 신규 상품을 기존 run에 누적 병합합니다.
4. division별 어노테이션 요청 큐가 기존 배치와 후속 배치를 순서대로 처리합니다.
5. 오래 걸리는 작업은 FastAPI background task로 실행합니다. 각 task는 서로 격리되어 하나가 실패해도 나머지가 계속 실행됩니다.

예를 들어 1~4번 상품 처리 중 1~7번 상품으로 재요청하면, 기존 상품의 완료·진행 상태를 유지한 채 5~7번을 추가합니다. 후속 배치에서는 이미 `awaiting_review` 이상으로 진행된 상품은 건너뛰고 실패 상품과 신규 상품만 실행합니다. 5~7번만 보내든 1~7번을 다시 보내든 결과는 같습니다. `product/created`는 payload의 상품 목록이 아니라 division 전체를 MongoDB에서 다시 조회해 분류하기 때문입니다.

어노테이션 큐의 division 워커는 토큰으로 소유권을 갖습니다. 워커가 비정상 종료하면 토큰을 반납하므로, 남은 배치는 다음 요청이 띄우는 새 워커가 순서대로 이어서 처리합니다. 워커가 중복으로 뜨지 않으면서도 큐가 영구히 멈추지 않습니다.

분기 기준:

| 조건 | 처리 |
| --- | --- |
| `isNew=1`, `status=2` | SAM3+DINO 자동 어노테이션 후 웹 검수 대기 |
| `isNew=1`, `status=3` | 웹 검수가 끝난 것으로 보고 segment 시작 |
| `isNew=0`, `status=2/3` | annotation/segment 생략 후 학습 준비 상태로 이동 |
| division 전체 `status=4` | YOLO 학습 시작 |
| division 전체 `status=6` | 모델 배포 시작 |
| division 전체 `status=7` | 배포 완료 |
| `status=8` 포함 | 검증 미통과, 수동 확인 필요 |

### `POST /v1/events/annotation/completed`

웹 검수가 완료된 상품을 segment 단계로 넘깁니다.

```json
{
  "HEADER": {
    "IF_ID": "IF_WEB_01",
    "IF_SYSID": "WEB-20260719-000001",
    "IF_HOST": "WEB",
    "IF_DATE": "20260719001000"
  },
  "DATA": {
    "division_idx": "DI17798460900133031",
    "product_idx": "P17835891041860813",
    "train_product_idx": "47",
    "is_cold": true
  }
}
```

동일한 `product_idx`가 여러 promotion 상품에 사용될 수 있으므로 중복 가능성이 있으면 `train_product_idx`를 반드시 전달해야 합니다.

### 테스트·복구용 API

- `POST /v1/test/run-annotation`: division의 자동 어노테이션 수동 실행
- `POST /v1/test/advance-pipeline`: segment·학습·배포 advance 수동 실행

운영 데이터와 MongoDB 상태를 변경할 수 있으므로 테스트 DB 또는 명확한 복구 상황에서만 사용합니다.

### HTTP 응답 코드

정상 접수는 HTTP `200`과 `DATA.result_cd=S`를 함께 반환합니다. 오류도 PNT `HEADER/DATA` 응답 형식은 유지하지만 실제 HTTP 상태 코드를 사용합니다.

| HTTP 상태 | 의미 |
| ---: | --- |
| `400` | 필수 필드 누락, JSON 또는 값 형식 오류 |
| `404` | division 또는 product를 찾지 못함 |
| `409` | 중복 product로 대상을 하나로 결정할 수 없음 |
| `500` | 처리되지 않은 서버 오류 또는 상태 갱신 실패 |
| `503` | MongoDB 설정 또는 접속 장애 |

오류 응답은 `DATA.result_cd=F`이며, `result_msg`에 누락 필드나 오류 원인이 포함됩니다. 호출자는 HTTP 상태와 `result_cd`를 모두 확인합니다.


## 7. SAM3 + Grounding DINO 자동 어노테이션

### 입력 구조

```text
img_download/
└── 47_BOTTLE_EXAMPLE_281ML_20260719_120000/
    ├── cam_0/
    │   ├── 000001.jpg
    │   └── ...
    └── cam_2/
        ├── 000001.jpg
        └── ...
```

카메라 디렉터리 안에 `images/`가 한 단계 더 있는 구조도 export 과정에서 지원합니다.

빈 냉장고의 고정 구조물 오탐을 줄이려면 division·카메라별 배경 참조 이미지를 둡니다. 문 열림·닫힘처럼 정상 배경 상태가 여러 개면 모두 등록할 수 있으며, 각 프레임에서 가장 가까운 배경을 자동으로 선택합니다.

```text
img_download/_backgrounds/
└── DI17846216186054023/
    └── cam_2/
        ├── door_closed.jpg
        └── door_open.jpg
```

API 실행은 기본적으로 `img_download/_backgrounds/<division_idx>/<camera>/`를 자동 조회합니다. 참조가 없는 division은 배경 필터만 생략하고 기존 공통 탐지 로직으로 처리합니다. 이 방식은 모델 가중치를 재학습하는 것이 아니라, 등록한 빈 배경과 다른 실제 전경만 DINO/SAM3 결과로 허용하는 negative-reference 필터입니다.

### 공통 탐지 방식

1. 상품 폴더명에서 숫자 prefix를 원본 class ID로 읽고 모든 프레임의 손·상품 DINO 후보를 배치 추론합니다.
2. 첫 번째 pass에서 영상을 전·후반으로 나누고 가림 여부를 함께 고려한 dual hand seed로 SAM3 Video Tracking을 수행합니다. 고정 frame chunk는 사용하지 않습니다.
3. pass 1의 프레임별 SAM3 손 mask 면적을 DINO 손 bbox 면적으로 나눈 fill ratio를 계산하고, DINO 결과와 상품 track을 run별 cache에 저장합니다.
4. 두 번째 pass는 DINO와 상품 cache를 재사용하며 fill ratio `0.45` 이상인 손 seed를 우선해 다시 추적합니다.
5. 상품은 `bottle`, `can`, `box`, `bag`, `cup`, `product package` 등 일반 형상 prompt를 사용하고 손은 전체 손 후보와 손가락·손바닥 보조 후보를 함께 사용합니다.
6. 손가락·손바닥 후보를 상품 근처에서 묶어 compact bbox를 만들고, SAM3 손 mask가 팔뚝 방향으로 과도하게 번지지 않도록 제한합니다.
7. 각 프레임에서 Video Tracking 결과와 DINO 후보를 이전 프레임 IoU, 중심 이동, 크기 변화로 결합합니다.
8. 배경 참조가 있으면 후보와 최종 mask의 실제 전경 비율을 검사하고, 화면 가장자리에 닿은 과대 손 bbox와 3프레임 이하의 짧은 손 탐지 구간을 제거합니다.
9. 손과 상품의 bbox·mask가 사실상 같으면 교차 클래스 중복으로 판단해 낮은 쪽을 제거하고, 가까워도 형상과 면적이 다르면 두 클래스를 유지합니다.
10. Detection bbox와 Segmentation polygon을 기록하며, 탐지하지 않은 배경 프레임도 빈 라벨 파일과 시각화 이미지를 남긴 뒤 웹 검수 COCO JSON을 생성합니다.

카메라와 상품 사이 거리에 따라 화면 점유율이 크게 달라지므로 프레임 대비 절대 면적, 너비·높이, 종횡비로 손이나 상품 후보를 제거하지 않습니다. `SAM3_DINO_MIN_AREA_RATIO`와 `SAM3_DINO_MAX_AREA_RATIO`는 절대 크기 제한이 아니라 SAM3 mask가 해당 DINO prompt bbox와 지나치게 다르게 번지는지를 확인하는 정합성 기준입니다.

특정 상품 번호나 45~49번 상품명에만 맞춘 분기는 사용하지 않습니다. 새 상품은 동일한 폴더명 규칙과 `cam_0`, `cam_2` 입력 구조만 맞추면 공통 로직으로 처리됩니다.

### API GPU 자동 선택

`POST /v1/events/product/created`로 시작된 어노테이션과 웹 검수 완료 후 실행되는 세그먼트는 같은 GPU 슬롯 풀을 공유합니다. NVML에서 `PIPELINE_ANNOTATION_GPUS`의 실제 여유 VRAM과 사용률을 조회하고, 작업 하나는 GPU 총 VRAM의 50% 이상이 실제로 비어 있을 때 한 슬롯을 예약합니다. 기본적으로 두 종류의 작업을 합쳐 GPU당 최대 2개이며 GPU 0, 1이 모두 충분히 비어 있으면 전체 최대 4개까지 실행할 수 있습니다. 예약 수가 적은 GPU를 먼저 고른 뒤 여유 VRAM, 사용률, 외부 프로세스 수 순으로 선택합니다. 세그먼트 subprocess에도 예약한 물리 GPU 번호를 전달합니다.

절대 최소 여유 VRAM은 `24576MiB`, 최대 사용률은 `80%`입니다. 두 번째 작업도 NVML상 50% 이상이 비어 있어야 시작하므로 첫 작업의 실사용량이 큰 경우에는 자동으로 대기합니다. 조건을 만족하는 GPU가 없으면 5초마다 다시 확인하고 최대 `ANNOTATION_GPU_SLOT_WAIT_SEC` 동안 대기합니다. 슬롯 선택, 대기 사유, 반환 상태는 기존 `annotation` phase 로그에 기록되며 세그먼트 작업은 `segment:` 접두사로 구분합니다. 슬롯은 성공·실패·예외에 관계없이 반환하며, 세그먼트 실패는 `failed` phase로 기록합니다. 외부 프로그램과 별도 API 프로세스는 이 예약 풀을 공유하지 않으므로 작업 시작 직전의 NVML 상태는 감지하지만, 완전히 동시에 시작되는 외부 작업까지 원자적으로 막지는 못합니다.

### 독립 실행

MongoDB 상태를 변경하지 않고 로컬 결과만 확인할 때 사용합니다.

```bash
docker compose exec pnt-api python \
  auto-labeling/sam3_with_dino/asis_pipeline.py \
  --img-root /app/img_download \
  --runs-root /app/runs/sam3_with_dino \
  --product 47_BOTTLE_EXAMPLE_281ML_20260719_120000 \
  --cams cam_0 cam_2 \
  --run-name manual_test_20260719 \
  --gpu 1 \
  --backend video \
  --background-reference-root /app/img_download/_backgrounds/DI17846216186054023 \
  --verbose
```

`--gpu 1`은 컨테이너에 노출된 물리 GPU 1만 해당 프로세스에 보이도록 설정합니다. 동시에 여러 작업을 실행하기 전에는 반드시 `nvidia-smi`로 메모리를 확인합니다.

### 결과 구조

```text
runs/sam3_with_dino/<run_name>/<product>/
├── cam_0/
│   ├── labels/                 # YOLO Detection 라벨, 빈 프레임은 빈 txt
│   ├── labels_seg/             # YOLO Segmentation 라벨
│   ├── visualized/             # 전체 프레임 시각화
│   ├── debug/detections.jsonl  # 프레임별 후보와 선택 근거
│   ├── summary.txt
│   └── run.log
├── cam_2/
│   └── ...
└── out/
    ├── annotation.json
    ├── annotations_<영문상품명>.json
    └── export_web_annotations.log
```

COCO JSON의 category는 다음처럼 변환됩니다.

- category `1`: `hand`
- category `2`: 해당 상품명

원본 프레임은 모두 `images`에 포함되고 탐지가 없는 배경 프레임은 annotation이 없는 이미지로 남습니다.

자동 어노테이션이 완료되면 MinIO의 `NewAnnotation` 아래에 웹 검수 패키지가 생성됩니다.

```text
NewAnnotation/<division>_<C|F>_<train_product_idx>_<product_idx>/
├── cam_0/
│   └── <상단 카메라 원본 이미지 전체>
├── cam_2/
│   └── <측면 카메라 원본 이미지 전체>
└── annotations_<영문상품명>.json
```

통합 COCO JSON의 `images[].file_name`은 원본 폴더명인 `cam_0`, `cam_2` 경로를 그대로 가리킵니다. 이미지를 먼저 모두 업로드하고 마지막에 JSON을 업로드하므로, 신규 작업은 불완전한 패키지가 웹 검수 대상으로 노출되지 않습니다.


### 일반화 품질 벤치마크

`benchmark.py`는 포장 유형별 상품을 개발군과 홀드아웃군으로 나누고 영상 시작·중간·끝의 연속 프레임을 동일하게 평가합니다.

```bash
docker run --rm \
  -v /data:/data:ro \
  -v /data/aipnt/pnt-api/auto-labeling/sam3_with_dino:/code:ro \
  -v /data/aipnt/pnt-api/runs/quality_benchmark:/out \
  --entrypoint python crk-0-pnt-api:latest \
  /code/benchmark.py prepare \
  --source-root /data/CRK_ewha/0_pnt/img_download \
  --reference-root /data/CRK_ewha/0_pnt/auto-labeling/sam2/results \
  --output-root /out/dataset \
  --max-products 10 --windows 3 --window-size 2
```

결과 평가:

```bash
docker run --rm \
  -v /data/aipnt/pnt-api/auto-labeling/sam3_with_dino:/code:ro \
  -v /data/aipnt/pnt-api/runs/quality_benchmark:/benchmark \
  --entrypoint python crk-0-pnt-api:latest \
  /code/benchmark.py evaluate \
  --benchmark-root /benchmark/dataset \
  --results-root /benchmark/results/<run_name>
```

기존 COCO 결과는 사람이 확정한 정답이 아니라 pseudo-reference일 수 있습니다. F1과 IoU는 변경 전후 비교용으로만 사용하고, 최종 판단에는 전체 시각화, 빈 프레임 오탐, 손·상품 관계와 홀드아웃 지표를 함께 사용합니다.

2026-07-19 공통 120프레임 비교에서는 object F1이 baseline `0.464`에서 최종 후보 `0.549`로, 개발군은 `0.371 -> 0.473`, 홀드아웃군은 `0.582 -> 0.638`로 개선됐습니다.


### 학습용 합성 데이터

검수 완료 라벨과 `productSeg/*.pkl`을 결합해 `divisionSynthesis/`에 학습 데이터를 생성합니다. 기본 설정은 전체 프레임에서 50%를 균등 간격으로 선택하고, step-by-step 레퍼런스의 강한 Gaussian·shake·streak blur를 segment에 적용합니다. 합성 조건을 만족하지 못한 선택 프레임도 `raw*.jpg/txt` 쌍으로 보존하며 빈 라벨은 background negative sample로 사용합니다.

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `SYNTH_AUGMENT_RATIO` | `0.5` | 합성 대상으로 선택할 원본 프레임 비율 |
| `SYNTH_STEP_BY_STEP` | `true` | 정렬된 전체 프레임에서 균등 간격으로 선택 |
| `SYNTH_INCLUDE_FAILED` | `true` | 합성 실패 프레임을 원본·라벨 쌍으로 유지 |
| `SYNTH_BLUR_PROFILE` | `strong` | `strong`은 레퍼런스 강도, `light`는 기존 운영 강도 |
| `SYNTH_BLUR_PROB` | `0.5` | 선택 segment에 blur를 적용할 확률 |

MinIO 업로드 전 이미지와 라벨의 stem 쌍을 검증합니다. `syn*`뿐 아니라 `raw*`도 같은 `divisionSynthesis/<division>_<train_idx>_<product>/` prefix에 업로드됩니다.

## 8. 파이프라인 상태

`logs/pipeline_state/<run_id>.json`은 MongoDB의 `training_status`와 별개인 AI 서버 orchestration 상태입니다.

상품 phase:

- `pending_annotation`: 자동 어노테이션 대기
- `awaiting_review`: 자동 어노테이션 완료, 웹 검수 대기
- `pending_segment`: segment 대기
- `segmenting`: segment 진행 중
- `ready_for_training`: 학습 준비 완료
- `skipped_existing`: 기존 상품이라 annotation/segment 생략
- `failed`: 작업 실패

run status:

- `in_progress`
- `training`
- `deployed`
- `completed`
- `failed`
- `superseded`: 레거시 상태. 누적 병합 도입 이전 상태 파일에만 남아 있고 더 이상 생성되지 않습니다
- `merged`: 같은 division의 활성 run에 병합된 후속 요청

같은 `division_idx`와 냉장/냉동 구분으로 새 요청이 들어오면 기존 활성 run을 중단하지 않습니다. 후속 요청의 상품 목록을 활성 run에 병합하고, 후속 `IF_SYSID` 상태 파일에는 `merged_into`로 실제 실행 run을 기록합니다. 서로 다른 division의 요청은 독립적으로 실행됩니다.

## 9. 로그와 Grafana

### 빠른 로그 확인

```bash
docker compose logs -f --tail=200 pnt-api
```

```bash
docker compose logs -f --tail=100 loki alloy
```

요청별 파이프라인 로그:

```bash
find logs/requests -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
  | sort -nr \
  | head
```

```bash
tail -f logs/requests/<요청_디렉터리>/pipeline.log
```

파이프라인 상태:

```bash
python -m json.tool logs/pipeline_state/<run_id>.json
```

### 실제 저장 위치

| 데이터 | 위치 |
| --- | --- |
| API stdout 원본 | Docker inspect의 `.LogPath` |
| 요청별 상세 로그 | `/data/aipnt/pnt-api/logs/requests/<request>/pipeline.log` |
| 파이프라인 상태 | `/data/aipnt/pnt-api/logs/pipeline_state/*.json` |
| Loki 데이터 | `/var/lib/docker/volumes/0_pnt_loki-data/_data` |
| Alloy 수집 상태 | `/var/lib/docker/volumes/0_pnt_alloy-data/_data` |

현재 API Docker 로그는 파일당 최대 `20MB`, 최대 `5`개로 회전합니다. Loki 보존 기간은 `720h`, 즉 30일입니다.

현재 컨테이너의 정확한 원본 로그 경로:

```bash
docker inspect crk-0-pnt-api --format '{{.LogPath}}'
```

### Grafana

- 주소: `https://aipnt.atcrk.co.kr/grafana`
- 대시보드: `CRK PNT API Structured Logs`
- 직접 경로: `/grafana/d/crk-pnt-api-structured-logs/crk-pnt-api-structured-logs`

대시보드 기능:

- 최근 5분 `2xx`, `4xx`, `5xx` 응답 수
- status class별 시계열
- `All`, `2xx`, `4xx`, `5xx`, `400`, `404`, `409`, `500`, `503` 필터
- 요청 payload, 응답 payload, 상태 코드, 처리 시간, `IF_SYSID`를 포함한 구조화 HTTP transaction 로그
- 전체 API stdout 로그
- Docker healthcheck는 계속 실행하지만 `/health` 요청은 Uvicorn access log와 구조화 로그에서 제외
- 과거 로그 열람 중 위치가 초기화되지 않도록 기본 자동 새로고침 비활성화

비밀번호, token, access key, Mongo URI 등 민감 필드는 구조화 로그에서 `***REDACTED***`로 마스킹됩니다.

#### 서버 시스템 모니터링

- 통합 화면: `/grafana/d/pnt-system-overview/pnt-server-system-overview`
- 상세 호스트 화면: `/grafana/d/pnt-node-exporter-full/pnt-node-exporter-full`
- 상세 GPU 화면: `/grafana/d/pnt-nvidia-dcgm/pnt-nvidia-dcgm-exporter`
- 데이터소스: `PNT System Metrics` (`uid: pnt-prometheus`)

수집 및 보존 구성:

- `node-exporter`: CPU, RAM, load, filesystem, disk I/O, network 지표
- `dcgm-exporter`: GPU 사용률, VRAM, 온도, 전력, clock 지표
- `prometheus`: 15초 간격 수집, 최대 30일 및 10GB 보존
- Prometheus UI/API는 호스트 loopback의 `127.0.0.1:9097`에만 공개
- 시계열은 Docker named volume `prometheus-data`에 저장

대시보드는 Grafana Labs의 [Node Exporter Full 1860](https://grafana.com/grafana/dashboards/1860-node-exporter-full/)과 NVIDIA가 권장하는 [DCGM Exporter 12239](https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/)를 현재 서버에 맞게 적용했습니다. JSON 원본은 `observability/dashboards`에 저장됩니다.

재기동 및 Grafana 재배포:

```bash
docker compose up -d node-exporter dcgm-exporter prometheus
python3 observability/provision-grafana-system.py
```

외부 Grafana의 영구 Prometheus 데이터소스 정의는 `observability/grafana-prometheus-datasource.yaml`이며, 호스트의 `/etc/grafana/provisioning/datasources/pnt-prometheus.yaml`에 설치됩니다.

## 10. 운영 점검 명령

```bash
docker compose ps
docker inspect crk-0-pnt-api --format '{{json .Config.Cmd}} {{.State.Health.Status}}'
curl -fsS http://127.0.0.1:2140/health
```

GPU 확인:

```bash
nvidia-smi
docker compose exec pnt-api python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

API 및 어노테이션 프로세스 확인:

```bash
ps -ef | grep -E 'uvicorn|run_product.py|auto_annotate.py' | grep -v grep
```

포트 확인:

```bash
ss -lntp | grep ':2140'
```

## 11. 문제 해결

### API가 시작되지 않을 때

```bash
docker compose ps
docker compose logs --tail=300 pnt-api
docker compose config
```

`.env` 누락, MongoDB/MinIO 설정, Python import 오류를 우선 확인합니다.

### annotation batch lock timeout

일반 `product/created` 재요청은 division별 직렬 큐에 들어가므로 이전 batch의 300초 lock timeout으로 유실되지 않습니다. 이 오류가 발생한다면 테스트·복구 코드가 큐를 거치지 않고 직접 batch를 실행했거나, 이전 버전의 컨테이너가 실행 중인지 확인합니다.

```bash
ps -ef | grep -E 'run_product.py|auto_annotate.py' | grep -v grep
nvidia-smi
```

실행 중인 작업과 `logs/pipeline_state`의 활성 run을 함께 확인합니다. 단순히 lock 파일이나 상태 JSON만 삭제해서는 안 됩니다.

### GPU OOM

- API는 NVML로 다른 프로세스의 VRAM과 GPU 사용률을 확인한 후 할당합니다.
- API 프로세스 내부에서는 동일 GPU에 여러 상품을 병렬 할당하지 않습니다.
- `PIPELINE_ANNOTATION_GPUS`와 실제 `CUDA_VISIBLE_DEVICES`를 확인합니다.
- 필요하면 `ANNOTATION_GPU_MIN_FREE_MEMORY_MIB`를 높이거나 `ANNOTATION_GPU_MAX_UTILIZATION`을 낮춥니다.
- OOM 이후 남은 worker가 있는지 확인하고 정상 종료 여부를 로그로 판단합니다.

다른 사용자의 GPU 프로세스는 임의로 종료하지 않습니다.

### Grafana에 로그가 보이지 않을 때

```bash
docker compose ps loki alloy
docker compose logs --tail=200 loki alloy
curl -fsS http://127.0.0.1:3100/ready
```

Alloy가 Docker socket을 읽을 수 있는지, `pnt-api` 서비스 label이 유지되는지 확인합니다.

## 12. 변경 원칙

- 자동 어노테이션 품질 기준은 모든 상품에 공통 적용합니다.
- 특정 상품명이나 특정 상품 번호 전용 threshold를 코드에 추가하지 않습니다.
- 품질 변경은 여러 상품과 `cam_0`, `cam_2`를 함께 평가한 뒤 반영합니다.
- 배경 프레임은 억지 탐지하지 않고 빈 라벨을 정상 데이터로 유지합니다.
- 웹 검수용 COCO JSON과 YOLO Detection/Segmentation 라벨의 class mapping을 함께 검증합니다.
- 운영 API 테스트에서 실제 MongoDB 상태를 변경하는 엔드포인트는 명시적인 대상 확인 없이 호출하지 않습니다.

