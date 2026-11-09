# 자동 어노테이션 운영 배포

진입점은 `auto-labeling/sam3_with_dino/annotation_pipeline.py`, 추론 패키지는 같은 경로의 `annotation_pipeline/`이다. 기존 AS-IS 두 번 실행 후 고정 손 검증·복원과 상품 국소 재검출을 수행하고 최종 검수 COCO를 생성한다.

현재 릴리스는 `auto-annotation-20261026.1`, 원 실험은 E027이다. 이번 릴리스는 이름·경로 정리이며 모델 바이트, 학습, 판정 임계값은 변경하지 않았다. 손은 최대 하나이며 기존 640×480 cam_0/cam_2 환경을 대상으로 한다.

냉동 상품의 상단 `cam_0/cam_1`처럼 `cam_1`이 포함되면 AS-IS 2-pass 뒤에 상단 프로필 후처리를 적용한다. 기존 손 후보 검증·복원·시간축 일관성·비손 제거 및 상품 국소 재검출은 재사용하고, 특정 매장의 참조 배경에 의존하는 스티커 제거만 생략한다. 원래 카메라 이름으로 COCO를 생성하며 `prepared.json`과 완료 표식에 `camera_profile: top`, `reference_background_filter: false`를 기록한다. `cam_0/cam_2` 상품은 참조 배경을 포함한 기존 후처리를 사용한다.

## 소스와 모델 준비

Git에 올릴 소스/설정 파일은 `Dockerfile.annotation`, `docker-compose.yml`, `gpu_scheduler.py`, `.gitignore`, 이 문서, `auto-labeling/sam3_with_dino/annotation_pipeline.py`, `asis_pipeline.py`, `requirements-annotation-cpu.txt`, `annotation_pipeline/*.py`, `model_releases/auto_annotation/20261026.1/manifest.json` 및 `README.md`다.

기존 `main.py`, `annotation_job.py`, AS-IS 엔진과 `requirements.txt`는 기존 저장소 파일을 사용한다. 별도 `e027_legacy_gpu_scheduler_v001.py`는 필요하지 않다. Dockerfile은 표준 `gpu_scheduler.py`를 복사하며 이번 변경에서 해당 파일을 실제 기존 실행 코드와 맞췄다. 차이는 로그 분류와 설명 두 곳뿐이다.

모델 자산 16개는 소스와 별도 배포한다. 해당 릴리스의 모델 아카이브를 저장소 루트에 풀어 `model_releases/auto_annotation/20261026.1/manifest.json`에 열거된 파일이 모두 있도록 한다. `Bundle`은 실행 때 모든 자산의 SHA256을 검증한다. 가중치·참조 이미지·외부 검출기 파일을 누락하면 동작하지 않는다. GitHub에 소스만 올렸다고 모델이 함께 배포되는 것은 아니다. 자산은 사내 아티팩트 저장소나 GitHub Release 등에서 별도로 관리할 수 있다.

## 이미지와 실행

`Dockerfile.annotation`은 기존 API 런타임 이미지를 기반으로 하며 기본 태그는 `crk-0-pnt-api:annotation-runtime-base-20261026`이다. 현재 검증한 기반 이미지 ID는 `sha256:9954db3ee5f72b3dd6ecd721841da6d314646e85b39282be34d079b18b4725aa`다. GitHub 소스에는 이 로컬 Docker 이미지가 포함되지 않는다. 다른 서버에서는 고정 기반 이미지를 전달/로드하거나 `--build-arg ANNOTATION_BASE_IMAGE=<고정된 레지스트리 이미지>`를 지정해야 한다. 기존 `Dockerfile`로 기반 이미지를 새로 만드는 경우 패키지 설치 결과와 추론을 다시 검증한다.

```bash
docker build -f Dockerfile.annotation -t crk-0-pnt-api:annotation-20261026.1 .
docker compose -p 0_pnt up -d --no-deps --no-build --pull never pnt-api
```

운영 교체 전 현재 이미지·compose·환경·모델을 백업하고 실행 중인 작업이 없음을 확인한다. 기존 데이터 볼륨과 모델 캐시는 보존한다. compose는 runner 경로와 모델 폴더를 설정하며, 모델은 읽기 전용 마운트다. CPU 추가 의존성은 `/opt/annotation_cpu_vendor`에 설치하고 CPU 자식 프로세스에서만 활성화한다. 전역 API/GPU 패키지를 덮어쓰지 않는다.

환경변수 `ANNOTATION_MODEL_DIR`와 `ANNOTATION_CPU_VENDOR`로 모델 및 CPU 의존성 경로를 지정할 수 있다. 기본값은 `/app/model_releases/auto_annotation/20261026.1` 및 `/opt/annotation_cpu_vendor`다.

새 결과의 후처리 근거는 각 상품 실행의 `annotation/`, 완료 표식은 `ANNOTATION_COMPLETE.json`에 기록한다. 기존 결과 파일은 변경하지 않는다. 실험 출처 E027은 모델 manifest와 이미지 라벨에서 확인한다.

상품별 실행 로그와 중간 캐시는 `{run_name}__annotation_{UTC시각(마이크로초 포함)}_{고유문자열}/`에 분리한다. 폴더는 원자적으로 생성하므로 같은 시각에 같은 요청의 여러 상품을 실행해도 충돌하지 않는다. AS-IS의 pass 로그·요약·캐시는 해당 폴더의 `asis/` 아래에 기록하며, 최종 결과 경로는 기존 `{run_name}/{상품폴더}/`를 사용한다. 단독 AS-IS 실행은 `--asis-work-dir`을 지정하지 않으면 기존 경로를 사용한다.

## 현재 서버의 원복 기록

현재 서버의 전체 변경·검증·백업·원복 명령은 개발 폴더 `yoona/yoona.md`의 D004 절에 통합 관리한다. 백업에는 운영 환경 비밀값이 포함되므로 GitHub 소스에 올리지 않는다. Docker 서버와 Jupyter의 loopback이 다르므로 상태 검사는 컨테이너 내부 `/health`와 서비스 네트워크를 기준으로 수행한다.
