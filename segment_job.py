"""
자동 세그먼트(PKL) job — `main.py`에서 웹 검수 완료(`annotation/completed`) 후 호출.

흐름:
  1) 이미지: `ANNOTATION_BASE_DIR/{foldername}` 로컬 재사용 → 없으면 MinIO `productImg/`
  2) 라벨: MinIO `productAnnotation/` (웹 검수 완료 YOLO txt)
     - `NewAnnotation/`(자동 어노테이션 초안)은 사용하지 않음
  3) `auto-segment/sam/auto-segment.sh` → SAM PKL → MinIO `productSeg/` 업로드

환경 변수:
- SEGMENT_BASE_DIR: 세그먼트 작업 루트 (기본 `./segment_work`)
- ANNOTATION_BASE_DIR: segment 시 이미지 재사용 경로 (annotation_job과 동일)
- MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY
- (선택) SEGMENT_SCRIPT: 미지정 시 `auto-segment/sam/auto-segment.sh`
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_SEGMENT_BASE_DIR = _PROJECT_ROOT / "segment_work"
_DEFAULT_ANNOTATION_BASE_DIR = _PROJECT_ROOT / "img_download"


def _find_actual_labels_dir(labels_dir: str) -> str:
    """cam0/cam_0 등 cam 디렉토리가 있는 실제 부모 폴더를 찾아 반환.
    MinIO 구조가 labels/train/cam0/ 처럼 중첩되어 있어도 올바른 경로를 찾습니다."""
    cam_names = {"cam0", "cam_0", "cam1", "cam_1", "cam2", "cam_2", "cam3", "cam_3"}
    for root, dirs, _ in os.walk(labels_dir):
        if any(d in cam_names for d in dirs):
            return root
    return labels_dir

try:
    from .job_log import run_shell_job
    from .request_log import get_request_context
except ImportError:
    from job_log import run_shell_job
    from request_log import get_request_context


def _parse_minio_folderpath(folderpath: str) -> Tuple[str, str]:
    """
    folderpath → (bucket, prefix) 분리.
    예: "/chaiimage/productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426/"
        → ("chaiimage", "productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426")
       "s3://chaiimage/productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426/"
        → ("chaiimage", "productImg/17_BOX_LOTTE_PEPERO_ALMOND_37G_20260129_175426")
    """
    clean_path = folderpath.strip()
    if clean_path.startswith("s3://"):
        clean_path = clean_path[len("s3://"):]
    clean_path = clean_path.strip("/")
    parts = clean_path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _find_existing_prefix(client, bucket: str, prefix: str) -> str:
    """prefix 아래 객체가 하나라도 있으면 해당 prefix를 반환."""
    clean_prefix = prefix.rstrip("/")
    objects = client.list_objects(bucket, prefix=clean_prefix + "/", recursive=True)
    for _ in objects:
        return clean_prefix
    return ""


def download_images_from_minio(
    train_product_idx: str,
    product_eng_name: str,
    local_dir: str,
    foldername: str = "",
    folderpath: str = "",
) -> int:
    """
    MinIO productImg 아래 이미지를 다운로드합니다.
    1) ProductsList.folderpath/foldername 기준 exact 경로 우선
    2) 없으면 train_product_idx_engname_* 형식 fallback
    하위 디렉토리 구조(cam_*/images/ 등)를 그대로 유지합니다.
    반환: 다운로드한 파일 수.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[segment_job] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        return 0

    try:
        from minio import Minio
    except ImportError:
        print("[segment_job] minio 라이브러리 없음: pip install minio", file=sys.stderr)
        return 0

    bucket = "chaiimage"
    exact_prefix = ""
    if folderpath:
        parsed_bucket, parsed_prefix = _parse_minio_folderpath(folderpath)
        if parsed_bucket:
            bucket = parsed_bucket
        exact_prefix = parsed_prefix
    elif foldername:
        exact_prefix = f"productImg/{foldername}"

    # train_product_idx_engname_* 형식으로 prefix 구성
    search_prefix = f"productImg/{train_product_idx}_{product_eng_name}_"

    if exact_prefix:
        print(f"[segment_job] 이미지 exact 검색: bucket={bucket}, prefix={exact_prefix}")
    else:
        print(f"[segment_job] 이미지 검색: bucket={bucket}, prefix={search_prefix}")
    
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    image_exts = (".jpg", ".jpeg", ".png")
    count = 0
    found_folder = None

    try:
        if exact_prefix:
            found_folder = exact_prefix.rstrip("/")
            objects = client.list_objects(bucket, prefix=found_folder + "/", recursive=True)
            object_names = [obj.object_name for obj in objects]
            if not object_names:
                print(f"[segment_job] exact prefix에 이미지가 없음: {exact_prefix}", file=sys.stderr)
                return 0

            print(f"[segment_job] 찾은 폴더: {found_folder}")
            for obj_name in object_names:
                name_lower = obj_name.lower()
                if not any(name_lower.endswith(ext) for ext in image_exts):
                    continue
                rel_path = obj_name[len(found_folder):].lstrip("/")
                dest = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                client.fget_object(bucket, obj_name, dest)
                count += 1
        else:
            # train_product_idx_engname_* 형식의 폴더 찾기
            objects = client.list_objects(bucket, prefix=search_prefix, recursive=False)
            for obj in objects:
                obj_name = obj.object_name
                # 폴더명 추출 (productImg/{foldername}/ 형태)
                # MinIO에서는 폴더가 /로 끝나거나, 또는 상위 경로를 추출
                if obj_name.endswith("/"):
                    found_folder = obj_name.rstrip("/")
                    break
                # 파일이면 상위 폴더명 추출
                parts = obj_name.split("/")
                if len(parts) >= 2:
                    found_folder = f"{parts[0]}/{parts[1]}"
                    break

            if not found_folder:
                print(f"[segment_job] MinIO에서 폴더를 찾지 못함: {search_prefix}", file=sys.stderr)
                return 0

            print(f"[segment_job] 찾은 폴더: {found_folder}")

            # 해당 폴더 아래 모든 이미지 다운로드
            objects = client.list_objects(bucket, prefix=found_folder + "/", recursive=True)
            for obj in objects:
                name_lower = obj.object_name.lower()
                if not any(name_lower.endswith(ext) for ext in image_exts):
                    continue
                # found_folder 이후 경로만 추출
                rel_path = obj.object_name[len(found_folder):].lstrip("/")
                dest = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                client.fget_object(bucket, obj.object_name, dest)
                count += 1
    except Exception as e:
        print(f"[segment_job] MinIO 이미지 다운로드 오류: {e}", file=sys.stderr)

    print(f"[segment_job] MinIO 이미지 다운로드 완료: {count}개 → {local_dir}")
    return count


def download_labels_from_minio(
    train_product_idx: str,
    product_eng_name: str,
    local_dir: str,
    foldername: str = "",
    product_idx: str = "",
    brunch_name: str = "",
) -> int:
    """
    MinIO productAnnotation에서 웹 검수 완료 라벨(txt)을 다운로드합니다.
    - productAnnotation: 웹 검수 완료 후 업로드되는 라벨 (segment에 사용해야 함)
    - NewAnnotation: 자동 어노테이션 결과 → segment에 사용하면 안 됨

    검색 순서:
    1) productAnnotation/{foldername}  (촬영일 기준 exact)
    2) productAnnotation/{train}_{engname}_*  (날짜 무관 와일드카드)
    cam_0/, cam_2/ 등 하위 구조 유지.
    반환: 다운로드한 파일 수.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[segment_job] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        return 0

    try:
        from minio import Minio
    except ImportError:
        print("[segment_job] minio 라이브러리 없음: pip install minio", file=sys.stderr)
        return 0

    bucket = "chaiimage"
    # productAnnotation/{train}_{engname}_* 와일드카드용 prefix
    search_prefix = f"productAnnotation/{train_product_idx}_{product_eng_name}_"

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    label_exts = (".txt",)
    count = 0

    def _download_txts_from_prefix(prefix: str) -> int:
        """prefix 아래 .txt 파일을 local_dir에 받아 개수 반환."""
        n = 0
        for obj in client.list_objects(bucket, prefix=prefix.rstrip("/") + "/", recursive=True):
            if not any(obj.object_name.lower().endswith(ext) for ext in label_exts):
                continue
            rel_path = obj.object_name[len(prefix):].lstrip("/")
            dest = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            client.fget_object(bucket, obj.object_name, dest)
            n += 1
        return n

    def _find_wildcard_folder() -> str:
        """productAnnotation/{train}_{engname}_* 로 날짜 무관하게 폴더 탐색.
        여러 폴더가 있으면 가장 최신(이름 기준 내림차순) 것을 사용."""
        folders = []
        for obj in client.list_objects(bucket, prefix=search_prefix, recursive=False):
            obj_name = obj.object_name
            folder = obj_name.rstrip("/") if obj_name.endswith("/") else "/".join(obj_name.split("/")[:2])
            if folder and folder not in folders:
                folders.append(folder)
        if not folders:
            return ""
        folders.sort(reverse=True)  # 날짜 내림차순 → 가장 최신
        return folders[0]

    try:
        # 1) productAnnotation/{foldername} — exact (촬영일 기준)
        if foldername:
            exact = f"productAnnotation/{foldername}"
            chk = _find_existing_prefix(client, bucket, exact)
            if chk:
                n = _download_txts_from_prefix(chk)
                if n > 0:
                    print(f"[segment_job] 찾은 라벨 폴더(exact): {chk} ({n}개)")
                    count = n
                else:
                    print(f"[segment_job] productAnnotation exact txt 0개, wildcard 시도: {chk}")

        # 2) productAnnotation/{train}_{engname}_* — 날짜 무관 와일드카드
        if count == 0:
            wc_folder = _find_wildcard_folder()
            if wc_folder:
                n = _download_txts_from_prefix(wc_folder)
                if n > 0:
                    print(f"[segment_job] 찾은 라벨 폴더(wildcard): {wc_folder} ({n}개)")
                    count = n
                else:
                    print(f"[segment_job] productAnnotation wildcard txt 0개: {wc_folder}", file=sys.stderr)
            else:
                print(f"[segment_job] MinIO에서 라벨 폴더를 찾지 못함: {search_prefix}", file=sys.stderr)

    except Exception as e:
        print(f"[segment_job] MinIO 라벨 다운로드 오류: {e}", file=sys.stderr)

    print(f"[segment_job] MinIO 라벨 다운로드 완료: {count}개 → {local_dir}")
    return count


def _count_local_images(root_dir: str) -> int:
    """이미지는 annotation 단계에서 받은 로컬 원본을 우선 재사용합니다."""
    root = Path(root_dir)
    if not root.is_dir():
        return 0

    image_exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    )


def run_segment_job(
    division_idx: str,
    product_info: dict,
    is_cold: bool = False,
) -> bool:
    """
    MinIO에서 이미지·검수 라벨을 받아 세그먼트(PKL) 합성을 실행합니다.

    이미지: `img_download/{foldername}` 재사용 → 없으면 `productImg/`.
    라벨: `productAnnotation/` 만 사용 (웹 검수 완료 후 업로드된 txt).

    product_info keys:
      - foldername, folderpath
      - product_idx, train_product_idx, product_eng_name
      - brunch_name (선택)
    """
    base = os.environ.get("SEGMENT_BASE_DIR", str(_DEFAULT_SEGMENT_BASE_DIR)).strip()
    if not base:
        print("[segment_job] SEGMENT_BASE_DIR 미설정", file=sys.stderr)
        return False

    foldername = (product_info.get("foldername") or "").strip()
    folderpath = (product_info.get("folderpath") or "").strip()
    product_idx = product_info.get("product_idx") or ""
    train_product_idx = str(product_info.get("train_product_idx") or "").strip()
    product_eng_name = (product_info.get("product_eng_name") or "").strip()
    brunch_name = (product_info.get("brunch_name") or "").strip()

    if not train_product_idx or not product_eng_name:
        print(f"[segment_job] train_product_idx 또는 product_eng_name이 비어있음: train_product_idx={train_product_idx}, product_eng_name={product_eng_name}", file=sys.stderr)
        return False

    # 작업 디렉토리 생성 (foldername 사용)
    if not foldername:
        foldername = f"{train_product_idx}_{product_eng_name}"
    
    local_work_root = os.path.join(base, foldername)
    local_labels_dir = os.path.join(local_work_root, "labels")
    os.makedirs(local_labels_dir, exist_ok=True)

    # 1) annotation 단계에서 이미 받은 이미지를 우선 재사용합니다.
    #    없을 때만 기존처럼 segment_work에 다운로드해서 복구/수동 실행 호환성을 유지합니다.
    annotation_base = os.environ.get("ANNOTATION_BASE_DIR", str(_DEFAULT_ANNOTATION_BASE_DIR)).strip()
    annotation_images_dir = os.path.join(annotation_base, foldername) if annotation_base else ""
    local_images_dir = annotation_images_dir
    img_count = _count_local_images(local_images_dir)

    if img_count > 0:
        print(f"[segment_job] annotation 이미지 재사용: {img_count}개 → {local_images_dir}")
    else:
        local_images_dir = os.path.join(local_work_root, "images")
        os.makedirs(local_images_dir, exist_ok=True)
        img_count = download_images_from_minio(
            train_product_idx,
            product_eng_name,
            local_images_dir,
            foldername=foldername,
            folderpath=folderpath,
        )
        if img_count == 0:
            print(f"[segment_job] MinIO에서 이미지를 받지 못함: {train_product_idx}_{product_eng_name}_*", file=sys.stderr)
            return False
        print(f"[segment_job] 이미지 fallback 다운로드 완료: {img_count}개")

    # 2) MinIO productAnnotation(웹 검수 완료 라벨) 다운로드
    label_count = download_labels_from_minio(
        train_product_idx,
        product_eng_name,
        local_labels_dir,
        foldername=foldername,
        product_idx=product_idx,
        brunch_name=brunch_name,
    )
    if label_count == 0:
        print(f"[segment_job] MinIO에서 라벨을 받지 못함: productAnnotation/{train_product_idx}_{product_eng_name}_*", file=sys.stderr)
        return False
    print(f"[segment_job] 라벨 다운로드 완료: {label_count}개")

    # MinIO 폴더 구조(labels/train/cam0/ 등)로 인해 cam 폴더가 중첩될 수 있으므로
    # 실제 cam0/ 등이 있는 부모 디렉토리를 찾아 LABELS_DIR로 사용
    adjusted_labels_dir = _find_actual_labels_dir(local_labels_dir)
    if adjusted_labels_dir != local_labels_dir:
        print(f"[segment_job] LABELS_DIR 조정: {local_labels_dir} → {adjusted_labels_dir}")
        local_labels_dir = adjusted_labels_dir

    # 3) 다운로드 확인
    print(f"[segment_job] 데이터 준비 완료: images={img_count}, labels={label_count}")
    print(f"[segment_job] 작업 디렉토리: {local_work_root}")

    # 4) auto-segment.sh 실행
    script_env = os.environ.get("SEGMENT_SCRIPT", "")
    if script_env:
        script_path = Path(script_env)
    else:
        repo_root = Path(__file__).resolve().parent
        script_path = repo_root / "auto-segment" / "sam" / "auto-segment.sh"
    
    if not script_path.exists():
        print(f"[segment_job] 스크립트 없음: {script_path}", file=sys.stderr)
        return False

    cwd = script_path.parent
    env = os.environ.copy()
    env["SEGMENT_BASE_DIR"] = base
    env["CLASS_NAME"] = foldername
    env["IS_COLD"] = "TRUE" if is_cold else "FALSE"
    env["IMAGES_DIR"] = local_images_dir
    env["LABELS_DIR"] = local_labels_dir
    env["TRAIN_PRODUCT_IDX"] = train_product_idx
    env["PRODUCT_ENG_NAME"] = product_eng_name
    env["DIVISION_IDX"] = division_idx
    env["MINIO_BASE_PREFIX"] = "productSeg"

    timeout_sec = int(os.environ.get("SEGMENT_JOB_TIMEOUT", "0")) or None

    try:
        ctx = get_request_context()
        job_log_dir = ctx.job_log_dir("segment") if ctx else None
        print(
            f"[segment_job] 실행: CLASS_NAME={foldername}, "
            f"product_idx={product_idx}, train_product_idx={train_product_idx}, "
            f"product_eng_name={product_eng_name}, division_idx={division_idx}, "
            f"is_cold={is_cold}, images_dir={local_images_dir}, labels_dir={local_labels_dir}, cwd={cwd}"
        )
        result = run_shell_job(
            ["/bin/bash", str(script_path)],
            job_type="segment",
            log_name=foldername,
            cwd=str(cwd),
            env=env,
            timeout=timeout_sec,
            log_prefix="[segment_job]",
            job_log_dir=job_log_dir,
        )
        if result.returncode != 0:
            print(f"[segment_job] 종료 코드 {result.returncode}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[segment_job] 타임아웃: {foldername}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[segment_job] 예외: {e}", file=sys.stderr)
        return False
