"""
모델 배포 job — `main.py` division 전원 status=6 시 `_maybe_advance_pipeline` → `deploy_job`.

흐름:
  1. YOLO run에서 `{division_idx}_{C|F}_*` 최신 `best.pt` 탐색
  2. MinIO `trained_model/{brunch_name}/{version}.pt` + `.onnx` 업로드
  3. `post_model_deploy_if14` — Mongo status=7, IF_14 POST
"""

import os
import sys
import io
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def _yolo_project_roots() -> List[Path]:
    """학습 run 디렉터리 후보 (ultralytics project 경로 차이 대응)."""
    roots: List[Path] = []
    env = os.getenv("YOLO_RUNS_DIR", "").strip()
    if env:
        roots.append(Path(env).expanduser())
    app_root = Path(__file__).resolve().parent
    repo_root = app_root.parent
    for base in (app_root, repo_root):
        roots.extend([
            base / "runs" / "detect" / "runs" / "train",
            base / "runs" / "train",
        ])
    seen = set()
    unique: List[Path] = []
    for path in roots:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _get_latest_best_pt(division_idx: str, is_cold: bool) -> Optional[Path]:
    """
    학습 완료된 best.pt 파일 경로를 찾습니다.
    """
    run_name_prefix = f"{division_idx}_{'C' if is_cold else 'F'}_"
    matching_runs: List[tuple] = []

    for project_path in _yolo_project_roots():
        if not project_path.is_dir():
            continue
        for run_dir in project_path.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith(run_name_prefix):
                continue
            best_pt = run_dir / "weights" / "best.pt"
            if best_pt.exists():
                matching_runs.append((run_dir.stat().st_mtime, best_pt, project_path))

    if not matching_runs:
        searched = ", ".join(str(p) for p in _yolo_project_roots())
        print(
            f"[deploy] best.pt 파일을 찾을 수 없습니다: prefix={run_name_prefix}* "
            f"(searched: {searched})"
        )
        return None

    matching_runs.sort(reverse=True)
    latest_best_pt = matching_runs[0][1]
    print(f"[deploy] 최신 best.pt 발견: {latest_best_pt} (root={matching_runs[0][2]})")
    return latest_best_pt


def _create_minio_client():
    """MinIO 클라이언트를 생성합니다."""
    try:
        from minio import Minio
    except ImportError:
        print("[deploy] minio 라이브러리 없음: pip install minio", file=sys.stderr)
        return None
    
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[deploy] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        return None
    
    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    except Exception as e:
        print(f"[deploy] MinIO 클라이언트 생성 실패: {e}", file=sys.stderr)
        return None


def _format_date_model_version(dt: Optional[datetime] = None) -> str:
    """
    학습/배포일 기준 버전 문자열. 예: 2026-05-21 → v26.5.21
    """
    dt = dt or datetime.now()
    return f"v{dt.year % 100}.{dt.month}.{dt.day}"


def _resolve_model_version_date() -> datetime:
    """MODEL_VERSION_DATE=YYYYMMDD 로 날짜 고정 가능 (테스트/재배포용)."""
    raw = os.getenv("MODEL_VERSION_DATE", "").strip()
    if raw:
        return datetime.strptime(raw, "%Y%m%d")
    return datetime.now()


def _get_next_model_version(client, bucket: str, brunch_name: str) -> str:
    """
    MinIO 기존 모델을 확인하고 배포 버전을 결정합니다.

    규칙: 학습(배포)일 기준 v{YY}.{M}.{D}
      - 예: 2026-05-21 → v26.5.21, 다음날 → v26.5.22
      - 같은 날 재배포 시 v26.5.21.1, v26.5.21.2 ...
    """
    base_version = _format_date_model_version(_resolve_model_version_date())
    prefix = f"trained_model/{brunch_name}/"

    existing_names = set()
    try:
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        for obj in objects:
            if obj.object_name.endswith(".pt"):
                existing_names.add(obj.object_name.split("/")[-1].replace(".pt", ""))
    except Exception as e:
        print(f"[deploy] MinIO 버전 조회 오류: {e}")
        print(f"[deploy] 다음 모델 버전(기본): {base_version}")
        return base_version

    if base_version not in existing_names:
        print(f"[deploy] 다음 모델 버전: {base_version}")
        return base_version

    suffix = 1
    while f"{base_version}.{suffix}" in existing_names:
        suffix += 1
    next_version = f"{base_version}.{suffix}"
    print(f"[deploy] 다음 모델 버전: {next_version} (같은 날 재배포)")
    return next_version


def _ensure_minio_path(client, bucket: str, brunch_name: str) -> bool:
    """
    MinIO에 trained_model/{brunch_name}/ 경로가 존재하는지 확인합니다.
    없으면 생성하지 않고 True 반환 (MinIO는 경로가 자동으로 생성됨).
    brunch_name 폴더가 이미 있으면 무시합니다.
    
    Args:
        client: MinIO 클라이언트
        bucket: 버킷 이름
        brunch_name: 브랜치 이름
    
    Returns:
        성공 여부
    """
    prefix = f"trained_model/{brunch_name}/"
    
    try:
        # trained_model/ 폴더 존재 확인 (빈 객체 생성으로 폴더 생성)
        trained_model_prefix = "trained_model/"
        objects = list(client.list_objects(bucket, prefix=trained_model_prefix, recursive=False))
        if not objects:
            # trained_model/ 폴더가 없으면 빈 객체로 생성
            print(f"[deploy] trained_model/ 폴더 생성: {bucket}/{trained_model_prefix}")
            try:
                # MinIO에서 폴더는 빈 객체로 표현됨
                client.put_object(bucket, trained_model_prefix, io.BytesIO(b""), length=0)
            except Exception as e:
                print(f"[deploy] trained_model/ 폴더 생성 시도 (이미 존재할 수 있음): {e}")
        
        # brunch_name 폴더 확인
        objects = list(client.list_objects(bucket, prefix=prefix, recursive=False))
        if objects:
            print(f"[deploy] brunch_name 폴더가 이미 존재합니다: {prefix} (무시)")
        else:
            print(f"[deploy] brunch_name 폴더 확인: {prefix}")
        
        return True
    except Exception as e:
        print(f"[deploy] MinIO 경로 확인 오류: {e}")
        return False


def _upload_file_to_minio(client, bucket: str, object_name: str, local_file: Path) -> bool:
    """로컬 파일 하나를 MinIO에 업로드합니다."""
    try:
        if not local_file.exists():
            print(f"[deploy] 파일이 없습니다: {local_file}")
            return False

        file_size = local_file.stat().st_size
        print(f"[deploy] MinIO 업로드 시작: {bucket}/{object_name} ({file_size} bytes)")
        client.fput_object(bucket, object_name, str(local_file))
        print(f"[deploy] MinIO 업로드 완료: {bucket}/{object_name}")
        return True
    except Exception as e:
        print(f"[deploy] MinIO 업로드 실패: {e}")
        return False


def _upload_model_to_minio(client, bucket: str, brunch_name: str, model_file: Path, model_version: str) -> bool:
    """
    모델 파일을 MinIO에 업로드합니다.
    
    Args:
        client: MinIO 클라이언트
        bucket: 버킷 이름
        brunch_name: 브랜치 이름
        model_file: 모델 파일 경로
        model_version: 모델 버전
    
    Returns:
        성공 여부
    """
    object_name = f"trained_model/{brunch_name}/{model_version}.pt"
    return _upload_file_to_minio(client, bucket, object_name, model_file)


def _convert_best_pt_to_onnx(best_pt: Path, output_path: Path) -> bool:
    """
    best.pt를 ONNX로 변환합니다.
    1) ultralytics Python API 시도
    2) 실패 시 yolo CLI 시도
    """
    if not best_pt.exists():
        print(f"[deploy] ONNX 변환 실패: 모델 파일 없음 {best_pt}")
        return False

    try:
        from ultralytics import YOLO

        model = YOLO(str(best_pt))
        exported_path = model.export(
            format="onnx",
            simplify=os.getenv("ONNX_SIMPLIFY", "true").strip().lower() == "true",
            opset=int(os.getenv("ONNX_OPSET", "12")),
        )
        exported = Path(str(exported_path))
        if not exported.exists():
            print(f"[deploy] ONNX 변환 결과 파일 없음: {exported}")
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported.replace(output_path)
        print(f"[deploy] ONNX 변환 완료(API): {output_path}")
        return True
    except Exception as api_err:
        print(f"[deploy] ONNX 변환 API 실패, CLI로 재시도: {api_err}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "yolo",
            "export",
            f"model={best_pt}",
            "format=onnx",
            f"opset={int(os.getenv('ONNX_OPSET', '12'))}",
            f"simplify={str(os.getenv('ONNX_SIMPLIFY', 'true')).lower()}",
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[deploy] ONNX 변환 CLI 실패: {proc.stderr.strip()}")
            return False
        cli_generated = best_pt.with_suffix(".onnx")
        if not cli_generated.exists():
            print(f"[deploy] ONNX 변환 CLI 결과 파일 없음: {cli_generated}")
            return False
        cli_generated.replace(output_path)
        print(f"[deploy] ONNX 변환 완료(CLI): {output_path}")
        return True
    except Exception as cli_err:
        print(f"[deploy] ONNX 변환 CLI 예외: {cli_err}")
        return False


def _upload_onnx_to_minio(client, bucket: str, brunch_name: str, onnx_file: Path, model_version: str) -> bool:
    """ONNX 파일을 같은 버전명으로 MinIO에 업로드합니다."""
    object_name = f"trained_model/{brunch_name}/{model_version}.onnx"
    return _upload_file_to_minio(client, bucket, object_name, onnx_file)


def run_deploy_job(division_idx: str, brunch_name: str, is_cold: bool = False) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    모델 배포 job 실행.
    MinIO에 trained_model/{brunch_name}/{version}.pt 형식으로 업로드합니다.
    
    Args:
        division_idx: division index
        brunch_name: 브랜치 이름
        is_cold: 냉동 여부
    
    Returns:
        (성공 여부, 모델 버전, 모델 파일명)
    """
    print(f"[deploy] 배포 job 시작: division_idx={division_idx}, brunch_name={brunch_name}, is_cold={is_cold}")
    
    # 1. best.pt 파일 찾기
    best_pt = _get_latest_best_pt(division_idx, is_cold)
    if not best_pt:
        return False, None, None
    
    # 2. MinIO 클라이언트 생성
    client = _create_minio_client()
    if not client:
        return False, None, None
    
    # 3. MinIO 버킷 설정
    bucket = os.getenv("MINIO_BUCKET", "chaiimage").strip()
    
    # 4. trained_model/{brunch_name}/ 경로 확인 (필요시 생성)
    if not _ensure_minio_path(client, bucket, brunch_name):
        return False, None, None
    
    # 5. 다음 모델 버전 결정
    model_version = _get_next_model_version(client, bucket, brunch_name)
    
    # 6. PT 모델 파일 MinIO에 업로드
    if not _upload_model_to_minio(client, bucket, brunch_name, best_pt, model_version):
        return False, None, None

    # 7. ONNX 변환 후 같은 버전명으로 업로드
    onnx_output = best_pt.parent / f"{model_version}.onnx"
    if not _convert_best_pt_to_onnx(best_pt, onnx_output):
        print("[deploy] ONNX 변환/업로드는 생략하고 PT 업로드만 완료 처리하지 않고 실패로 반환합니다.")
        return False, None, None
    if not _upload_onnx_to_minio(client, bucket, brunch_name, onnx_output, model_version):
        return False, None, None

    model_filename = f"{model_version}.pt"
    print(
        f"[deploy] 배포 job 완료: "
        f"{bucket}/trained_model/{brunch_name}/{model_version}.pt, "
        f"{bucket}/trained_model/{brunch_name}/{model_version}.onnx"
    )

    return True, model_version, model_filename
