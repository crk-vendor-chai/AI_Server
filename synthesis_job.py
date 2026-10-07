"""
합성 Job — `training_job.py` 학습 직전에 PKL 기반 이미지·라벨 합성.

흐름:
  1. MinIO `productSeg/` PKL 다운로드 → segment bank
  2. MinIO `productImg/`, `productAnnotation/` (검수 완료 라벨) 다운로드
  3. 규칙 기반 합성 (`apply_segment_shake_blur` 등)
  4. MinIO `divisionSynthesis/{division}_{train}_{eng}/` 업로드

Blur 정책:
  - 배치에 성공한 synthetic segment 중 정확히 절반에만 blur 적용
  - blur 강도는 `synthesis_job_step_by_step_half_blur.py`와 동일
  - 그 외 MinIO/카메라/라벨/class-id/raw-fallback 로직은 Docker 버전을 유지
"""
import os
import re
import sys
import cv2
import pickle
import random
import shutil
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    from .class_schema import encode_synthetic_class_id
except ImportError:
    from class_schema import encode_synthetic_class_id

SYNTH_BLUR_PROB = float(os.getenv("SYNTH_BLUR_PROB", "0.5"))

# pkl 안의 카메라 데이터 키 (cam_0, cam_1, cam_2 …).
# cam_0 = top, 그 외 = side 로 해석한다.
_CAM_KEY_RE = re.compile(r"^cam_(\d+)$")


def _shake_frame_average(seg: np.ndarray, n_samples: int, max_shift: int) -> np.ndarray:
    """짧은 위치 흔들림을 여러 프레임 평균으로 재현."""
    h, w = seg.shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(n_samples):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(seg, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        acc += shifted
    return (acc / n_samples).astype(np.uint8)


def _elongated_motion_kernel(length: int, thickness: int, angle_deg: float) -> np.ndarray:
    length = max(3, length | 1)
    thickness = max(1, thickness | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    cy = length // 2
    half_t = thickness // 2
    kernel[cy - half_t : cy + half_t + 1, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel /= kernel.sum() + 1e-6
    return kernel


def _apply_pull_motion_streak(seg: np.ndarray, length: int, angle_deg: float, weight: float) -> np.ndarray:
    thickness = 3 if length >= 7 else 1
    streak = cv2.filter2D(seg, -1, _elongated_motion_kernel(length, thickness, angle_deg))
    return cv2.addWeighted(streak, weight, seg, 1.0 - weight, 0)


def _apply_light_gaussian_blur(seg: np.ndarray) -> np.ndarray:
    """step_by_step_half_blur와 동일한 Gaussian blur 강도."""
    k = random.choice([3, 5])
    sigma = random.uniform(0.6, 1.4)
    return cv2.GaussianBlur(seg, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_segment_shake_blur(
    seg: np.ndarray,
    prob: float = SYNTH_BLUR_PROB,
) -> np.ndarray:
    """step_by_step_half_blur와 동일한 강한 Gaussian + shake + motion blur.

    합성 루프에서는 blur 대상으로 선정된 segment에 ``prob=1.0``으로 호출한다.
    mask와 bbox는 변경하지 않는다.
    """
    if random.random() > prob:
        return seg

    out = _apply_light_gaussian_blur(seg)

    h, w = out.shape[:2]
    seg_diag = int(np.hypot(h, w))

    # 위치 흔들림: 최대 3~7px, 4~7개 프레임 평균
    max_shift = max(3, min(7, seg_diag // 35))
    shaken = _shake_frame_average(
        out,
        n_samples=random.randint(4, 7),
        max_shift=max_shift,
    )

    # 물체 크기에 따라 9~21px motion streak
    streak_len = max(9, min(21, seg_diag // 8))
    if streak_len % 2 == 0:
        streak_len += 1

    angle = random.uniform(0, 360)
    out = _apply_pull_motion_streak(
        shaken,
        streak_len,
        angle,
        weight=random.uniform(0.50, 0.75),
    )

    # 50% 확률로 두 번째 motion streak 추가
    if random.random() < 0.50:
        angle2 = angle + random.uniform(-10, 10)
        len2 = random.choice([5, 7, 9])
        out = _apply_pull_motion_streak(
            out,
            len2,
            angle2,
            weight=random.uniform(0.20, 0.40),
        )

    return out


def _create_minio_client():
    """MinIO 클라이언트 생성"""
    try:
        from minio import Minio
    except ImportError:
        print("[synthesis] minio 라이브러리 없음: pip install minio", file=sys.stderr)
        return None
    
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    secure = os.environ.get("MINIO_SECURE", "false").strip().lower() == "true"
    if not (endpoint and access_key and secret_key):
        print("[synthesis] MinIO env 누락: MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY", file=sys.stderr)
        return None
    
    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    except Exception as e:
        print(f"[synthesis] MinIO 클라이언트 생성 실패: {e}", file=sys.stderr)
        return None


def _download_pkl(client, bucket: str, train_product_idx: str, product_eng_name: str, dest_dir: Path) -> Optional[Path]:
    """
    MinIO productSeg/에서 pkl 다운로드.
    파일명 형식: {train_product_idx}_{product_eng_name}_{날짜}.pkl
    가장 최신 파일 선택.
    """
    prefix = "productSeg/"
    search_prefix = f"{train_product_idx}_{product_eng_name}_"
    
    try:
        matching_objects = []
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            obj_name = obj.object_name
            filename = obj_name.split("/")[-1]
            if filename.startswith(search_prefix) and filename.endswith(".pkl"):
                matching_objects.append((obj.last_modified, obj_name))
        
        if not matching_objects:
            print(f"[synthesis] pkl 파일 없음: {bucket}/{prefix}{search_prefix}*.pkl")
            return None
        
        # 최신 파일 선택
        matching_objects.sort(reverse=True)
        latest_obj_name = matching_objects[0][1]
        
        dest_file = dest_dir / Path(latest_obj_name).name
        client.fget_object(bucket, latest_obj_name, str(dest_file))
        print(f"[synthesis] pkl 다운로드 완료: {latest_obj_name} -> {dest_file}")
        return dest_file
    except Exception as e:
        print(f"[synthesis] pkl 다운로드 오류: {e}", file=sys.stderr)
        return None


def _extract_segment_list(cam_data):
    if isinstance(cam_data, dict):
        segments = cam_data.get("segments")
        return segments if isinstance(segments, list) else []
    if isinstance(cam_data, list):
        return cam_data
    return []


def _resolve_annotation_prefix(client, bucket: str, foldername: str,
                               train_product_idx: str, product_eng_name: str) -> Optional[str]:
    """검수 라벨(productAnnotation) 폴더를 찾는다.

    foldername 은 촬영일 기준이라 검수일로 생성된 라벨 폴더와 어긋난다.
    1) foldername 정확 일치를 먼저 시도하고
    2) 없으면 {train_product_idx}_{eng_name}_ 로 시작하는 폴더 중 최신을 쓴다.
    반환값은 프리픽스(마지막 / 제외) 또는 None.
    """
    exact = f"productAnnotation/{foldername}"
    try:
        if any(client.list_objects(bucket, prefix=exact + "/", recursive=True)):
            return exact
    except Exception as e:
        print(f"[synthesis] 라벨 폴더 조회 오류: {e}", file=sys.stderr)

    if not train_product_idx or not product_eng_name:
        return None

    search = f"productAnnotation/{train_product_idx}_{product_eng_name}_"
    candidates = set()
    try:
        for obj in client.list_objects(bucket, prefix=search, recursive=True):
            rest = obj.object_name[len("productAnnotation/"):]
            top = rest.split("/", 1)[0]
            if top:
                candidates.add(f"productAnnotation/{top}")
    except Exception as e:
        print(f"[synthesis] 라벨 폴더 탐색 오류: {e}", file=sys.stderr)
        return None

    if not candidates:
        return None
    # 폴더명 끝에 YYYYMMDD_HHMMSS 가 붙으므로 사전순 최대 = 최신
    return max(candidates)


def _load_segment_bank(product_list: list, pkl_dir: Path, client, bucket: str) -> dict:
    """
    각 product의 pkl을 다운로드하여 segment bank 구축.
    
    반환: {'top': {train_product_idx: [segments]}, 'side': {train_product_idx: [segments]}}
    """
    bank = {'top': defaultdict(list), 'side': defaultdict(list)}
    incomplete = []   # top 또는 side 가 비어 있는 상품
    missing = []      # pkl 자체를 받지 못한 상품

    for product in product_list:
        train_product_idx = str(product.get("train_product_idx") or "").strip()
        product_eng_name = (product.get("product_eng_name") or "").strip()

        if not train_product_idx or not product_eng_name:
            print(f"[synthesis] train_product_idx 또는 product_eng_name 누락: {product}")
            continue

        pkl_file = _download_pkl(client, bucket, train_product_idx, product_eng_name, pkl_dir)
        if not pkl_file:
            missing.append(train_product_idx)
            continue

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # pkl 은 실제 카메라 이름(cam_0, cam_1, cam_2 …)을 키로 갖는다.
            # cam_0 = top, 그 외 cam_N = side 로 해석한다.
            # 리그마다 side 카메라 번호가 달라(_C=cam_2, _F=cam_1) 번호를 고정하지 않는다.
            top_segments = _extract_segment_list(data.get("cam_0"))
            side_segments = []
            for cam_key in sorted(k for k in data if _CAM_KEY_RE.match(str(k))):
                if cam_key == "cam_0":
                    continue
                side_segments.extend(_extract_segment_list(data.get(cam_key)))

            if top_segments:
                bank['top'][int(train_product_idx)].extend(top_segments)
            if side_segments:
                bank['side'][int(train_product_idx)].extend(side_segments)
            
            print(f"[synthesis] segment 로드: train_product_idx={train_product_idx}, "
                  f"top={len(top_segments)}, side={len(side_segments)}")

            # top/side 한쪽이라도 비면 그 상품은 합성에 온전히 쓰이지 못한다.
            # 조용히 넘어가면 결과가 0개로 끝난 뒤에야 알게 되므로 여기서 드러낸다.
            if not top_segments or not side_segments:
                incomplete.append(
                    f"{train_product_idx}(top={len(top_segments)},side={len(side_segments)})"
                )
                print(
                    f"[synthesis] 경고: segment 부족 — train_product_idx={train_product_idx}, "
                    f"top={len(top_segments)}, side={len(side_segments)}, pkl={pkl_file.name}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[synthesis] pkl 로드 실패: {pkl_file}, {e}", file=sys.stderr)

    print(
        f"[synthesis] segment bank 요약: 상품={len(product_list)}, "
        f"top={len(bank['top'])} classes, side={len(bank['side'])} classes, "
        f"pkl 없음={len(missing)}, segment 부족={len(incomplete)}"
    )
    if missing:
        print(f"[synthesis] 경고: pkl 미확보 상품 {missing}", file=sys.stderr)
    if incomplete:
        print(f"[synthesis] 경고: segment 부족 상품 {incomplete}", file=sys.stderr)

    return bank


# =========================================================
# 합성 유틸리티 (0105_synthesize.py 기반)
# =========================================================

def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    xc, yc = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    return [
        max(0, int(xc - bw / 2)),
        max(0, int(yc - bh / 2)),
        min(img_w, int(xc + bw / 2)),
        min(img_h, int(yc + bh / 2)),
    ]


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def compute_box_overlap(new_box, existing_box):
    """IoU + 기존/신규 bbox 각각의 겹침 비율 반환."""
    x1, y1, x2, y2 = new_box
    X1, Y1, X2, Y2 = existing_box

    ix1 = max(x1, X1)
    iy1 = max(y1, Y1)
    ix2 = min(x2, X2)
    iy2 = min(y2, Y2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0, 0.0, 0.0

    area_new = (x2 - x1) * (y2 - y1)
    area_old = (X2 - X1) * (Y2 - Y1)
    union = area_new + area_old - inter

    iou = inter / (union + 1e-6)
    overlap_old = inter / (area_old + 1e-6)
    overlap_new = inter / (area_new + 1e-6)
    return iou, overlap_old, overlap_new


def boxes_overlap_exceeds(box_a, box_b, max_overlap: float = 0.2) -> bool:
    """IoU 또는 어느 한쪽 bbox 면적의 max_overlap 초과 겹침이면 True."""
    iou, overlap_old, overlap_new = compute_box_overlap(box_a, box_b)
    return (
        iou > max_overlap
        or overlap_old > max_overlap
        or overlap_new > max_overlap
    )


def get_view(path_like):
    """이미지 경로에서 view 추출 (top/side).

    cam_0 은 top, 그 외 카메라(cam_1, cam_2, ...)는 side 로 본다.
    리그마다 side 카메라 번호가 달라(_C=cam_2, _F=cam_1) 번호를 고정하지 않는다.
    """
    path_lower = str(path_like).replace("\\", "/").lower()
    if "top" in path_lower:
        return "top"
    if "side" in path_lower:
        return "side"
    for part in path_lower.split("/"):
        matched = _CAM_DIR_RE.match(part)
        if matched:
            return "top" if int(matched.group(1)) == 0 else "side"
    return None


_CAM_DIR_RE = re.compile(r"^cam[_-]?(\d+)$")


def _camera_stem_key(path_like):
    """카메라 번호 + 파일명(stem)으로 매칭.

    이미지는 `cam_0/`, 라벨은 `cam0/`처럼 표기가 다르고 중간 경로도 다르므로
    상대경로 매칭(_rel_pair_key)이 실패한다. 이 키는 표기 차이를 흡수한다.
    cam_0/cam_1/cam_2 등 번호에 관계없이 동작한다.
    """
    path = Path(path_like)
    cam_name = None
    for part in path.parts:
        matched = _CAM_DIR_RE.match(part.lower())
        if matched:
            cam_name = f"cam_{int(matched.group(1))}"
            break
    return cam_name, path.stem


def _rel_pair_key(path_like, root: Path) -> Optional[str]:
    """images/ 또는 labels/ 기준 상대경로로 매칭 (cam_0/frame_000178)."""
    path = Path(path_like)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    return str(rel.parent / rel.stem).replace("\\", "/")


def get_bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return [x, y, x + w, y + h]


def rotate_image_and_mask(img, mask, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    img_r = cv2.warpAffine(img, M, (nw, nh))
    mask_r = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST)
    return img_r, mask_r


def calculate_union_occupancy(img_shape, bboxes):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 1
    return np.sum(mask) / (h * w + 1e-6)


def paste_segment(bg, seg, mask, x, y):
    h, w = seg.shape[:2]
    roi = bg[y:y+h, x:x+w]
    alpha = (mask / 255.0)[:, :, None]
    blended = seg * alpha + roi * (1 - alpha)
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)
    lb = get_bbox_from_mask(mask)
    if lb is None:
        return False, None
    return True, [x+lb[0], y+lb[1], x+lb[2], y+lb[3]]


def find_valid_position(img_shape, seg_size, existing_bboxes, view, max_iou=0.20, max_occlusion=0.2, max_attempts=100, side_left_margin=0.3, top_bottom_margin=0.4):
    h, w = img_shape[:2]
    sh, sw = seg_size

    if view == "side":
        xmin = int(w * side_left_margin)
        xmax = w - sw
        ymin = 0
        ymax = h - sh
    else:  # top
        xmin = 0
        xmax = w - sw
        ymin = 0
        ymax = int(h * (1 - top_bottom_margin)) - sh

    if xmax < xmin or ymax < ymin:
        return None

    for _ in range(max_attempts):
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        cand = [x, y, x+sw, y+sh]
        valid = True
        for b in existing_bboxes:
            if boxes_overlap_exceeds(cand, b, max_overlap=max_occlusion):
                valid = False
                break
        if valid:
            return x, y
    return None


def synthesize_on_images(
    segment_bank: dict,
    image_paths: list,
    label_paths: list,
    output_img_dir: Path,
    output_lbl_dir: Path,
    augment_ratio: float = 0.5,
    step_by_step: bool = False,
    include_failed: bool = False,
    min_products_side: int = 4,
    max_products_side: int = 10,
    min_products_top: int = 1,
    max_products_top: int = 3,
    scale_range_side: tuple = (0.7, 1.2),
    scale_range_top: tuple = (0.8, 1.0),
    max_occupancy_rate: float = 0.8,
    max_segment_area_ratio: float = 0.15,
    hand_class: int = 0,
    img_root: Optional[Path] = None,
    lbl_root: Optional[Path] = None,
):
    """
    원본 이미지에 segment 합성.
    
    Args:
        segment_bank: {'top': {class_id: [segments]}, 'side': {class_id: [segments]}}
        image_paths: 원본 이미지 경로 리스트
        label_paths: 원본 라벨 경로 리스트 (이미지와 1:1 대응)
        output_img_dir: 합성 이미지 저장 디렉토리
        output_lbl_dir: 합성 라벨 저장 디렉토리
        augment_ratio: 합성에 사용할 이미지 비율 (0.0~1.0)
        step_by_step: True면 augment_ratio만큼의 이미지를 무작위(random.sample) 대신
            경로 정렬 후 균등 간격 인덱스로 순차 선택한다.
            예) 이미지 10장, augment_ratio=0.5 -> index 0,2,4,6,8 선택.
            (정렬은 rglob() 등 파일시스템 순회 순서가 실행마다 달라질 수 있어
            "순차 선택"이 매번 같은 이미지를 가리키도록 하기 위함)
        include_failed: True면 targets로 선택됐지만 합성 조건(뷰 판별, segment bank 존재,
            이미지 로드, 기존 라벨 존재, occupancy 한도, needed>0, 세그먼트 배치 성공)을
            만족하지 못해 원래는 그냥 skip되던 이미지를, 원본 그대로(라벨도 원본 그대로,
            없으면 빈 라벨) output_img_dir/output_lbl_dir에 저장한다.
            즉 targets(=augment_ratio 비율)의 각 이미지가 예외 없이 하나씩
            output에 반영되어 "50%를 채운다."
            False(기본값)면 기존과 동일하게 조건 미달 이미지는 버려진다.
        ...
    
    반환: 생성된 합성 이미지 개수
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지와 라벨 매칭 (상대경로 우선, 실패 시 cam+stem 폴백)
    #
    # MinIO의 이미지/라벨은 디렉터리 구조가 다르다.
    #   이미지  productImg/{상품}/cam_0/000000.jpg
    #   라벨    productAnnotation/{상품}/labels/train/{brunch}/cam0/000000.txt
    # 그래서 상대경로 키는 서로 어긋난다. 라벨을 두 키로 색인해 두고
    # 조회가 실패했을 때 폴백 키로 다시 찾는다.
    label_map = {}
    for lbl_path in label_paths:
        rel_key = _rel_pair_key(lbl_path, lbl_root) if lbl_root else None
        if rel_key is not None:
            label_map.setdefault(rel_key, lbl_path)
        label_map.setdefault(_camera_stem_key(lbl_path), lbl_path)

    img_to_lbl = {}
    for img_path in image_paths:
        rel_key = _rel_pair_key(img_path, img_root) if img_root else None
        lbl_path = label_map.get(rel_key) if rel_key is not None else None
        if lbl_path is None:
            lbl_path = label_map.get(_camera_stem_key(img_path))
        img_to_lbl[img_path] = lbl_path

    matched_count = sum(1 for v in img_to_lbl.values() if v is not None)
    print(f"[synthesis] 이미지-라벨 매칭: {matched_count}/{len(img_to_lbl)}")

    # augment_ratio만큼 선택 (step_by_step=True면 무작위 대신 균등 간격 순차 선택)
    if not 0.0 <= augment_ratio <= 1.0:
        raise ValueError(f"augment_ratio must be between 0 and 1: {augment_ratio}")
    all_keys = sorted(img_to_lbl) if step_by_step else list(img_to_lbl)
    n_select = min(len(all_keys), int(len(all_keys) * augment_ratio))
    if step_by_step and n_select:
        step = len(all_keys) / n_select
        targets = [all_keys[int(index * step)] for index in range(n_select)]
    elif step_by_step:
        targets = []
    else:
        targets = random.sample(all_keys, n_select)
    
    syn_count = 0
    raw_count = 0
    syn_counter = 1

    def _save_raw_fallback(image: Optional[np.ndarray], src_img_path: Path, src_lbl_path: Optional[Path]) -> bool:
        """include_failed=True일 때, 합성 조건을 만족하지 못한 이미지를 원본 그대로 저장한다.
        라벨은 원본 파일 내용을 그대로 복사하고(있으면), 없으면 빈 라벨(negative sample)로 저장한다.
        파일명은 syn 대신 raw 접두사를 써서 실제 합성 결과와 구분할 수 있게 한다.
        """
        nonlocal syn_counter, raw_count
        raw_img_name = f"raw{syn_counter:05d}.jpg"
        raw_lbl_name = f"raw{syn_counter:05d}.txt"
        try:
            if image is not None:
                cv2.imwrite(str(output_img_dir / raw_img_name), image)
            else:
                loaded = cv2.imread(str(src_img_path))
                if loaded is not None:
                    cv2.imwrite(str(output_img_dir / raw_img_name), loaded)
                else:
                    # cv2로도 못 읽는 경우 파일 바이트 그대로 복사
                    shutil.copy2(src_img_path, output_img_dir / raw_img_name)
        except Exception as e:
            print(f"[synthesis] raw fallback 이미지 저장 실패: {src_img_path} ({e})", file=sys.stderr)
            return False

        if src_lbl_path is not None and src_lbl_path.exists():
            shutil.copy2(src_lbl_path, output_lbl_dir / raw_lbl_name)
        else:
            (output_lbl_dir / raw_lbl_name).write_text("", encoding="utf-8")

        raw_count += 1
        syn_counter += 1
        return True

    for img_path in targets:
        lbl_path = img_to_lbl[img_path]
        view = get_view(img_path)
        if view not in ["top", "side"]:
            if include_failed:
                _save_raw_fallback(None, img_path, lbl_path)
            continue
        if not segment_bank[view]:
            if include_failed:
                _save_raw_fallback(None, img_path, lbl_path)
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            if include_failed:
                _save_raw_fallback(None, img_path, lbl_path)
            continue
        h, w = img.shape[:2]
        
        existing_bboxes = []
        existing_lines = []
        product_cnt = 0
        
        if lbl_path and lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    bbox = yolo_to_xyxy(parts[1], parts[2], parts[3], parts[4], w, h)
                    existing_bboxes.append(bbox)
                    existing_lines.append(line.strip())
                    if cls != hand_class:
                        product_cnt += 1

        if not existing_bboxes:
            if include_failed:
                _save_raw_fallback(img, img_path, lbl_path)
            continue

        if calculate_union_occupancy(img.shape, existing_bboxes) > max_occupancy_rate:
            if include_failed:
                _save_raw_fallback(img, img_path, lbl_path)
            continue
        
        target_n = random.randint(
            min_products_top if view == "top" else min_products_side,
            max_products_top if view == "top" else max_products_side
        )
        
        needed = max(0, target_n - product_cnt)
        if needed == 0:
            if include_failed:
                _save_raw_fallback(img, img_path, lbl_path)
            continue
        
        aug = img.copy()
        new_lines = []
        cur_boxes = existing_bboxes[:]
        pending_pastes = []  # 배치 성공 segment를 모은 뒤 정확히 절반에만 blur를 적용한다.

        for _ in range(needed):
            # 랜덤 class_id 선택
            available_classes = list(segment_bank[view].keys())
            if not available_classes:
                break
            cls_id = random.choice(available_classes)

            # 랜덤 segment 선택
            seg_data = random.choice(segment_bank[view][cls_id])

            seg = seg_data["segment"]
            mask = seg_data["mask"].astype(np.uint8)

            # rotation
            angle = random.randint(-180, 180)
            seg, mask = rotate_image_and_mask(seg, mask, angle)

            # scale
            scale = random.uniform(*(scale_range_top if view == "top" else scale_range_side))
            sh, sw = int(seg.shape[0] * scale), int(seg.shape[1] * scale)
            if sh <= 0 or sw <= 0:
                continue

            if sh * sw > h * w * max_segment_area_ratio:
                continue

            seg = cv2.resize(seg, (sw, sh))
            mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

            # find position
            pos = find_valid_position(aug.shape, (sh, sw), cur_boxes, view)
            if pos is None:
                continue

            # blur 여부는 배치 성공 여부와 분리한다. 실제 픽셀 합성은 모든 배치가
            # 확정된 뒤 수행하고, 성공 segment 수를 기준으로 정확히 절반만 blur한다.
            lb = get_bbox_from_mask(mask)
            if lb is None:
                continue
            x, y = pos
            bbox = [x + lb[0], y + lb[1], x + lb[2], y + lb[3]]

            if any(boxes_overlap_exceeds(bbox, b) for b in cur_boxes):
                continue

            cur_boxes.append(bbox)
            xc, yc, bw, bh = xyxy_to_yolo(*bbox, w, h)
            synthetic_id = encode_synthetic_class_id(cls_id)
            new_lines.append(f"{synthetic_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            pending_pastes.append({"seg": seg, "mask": mask, "x": x, "y": y})

        # step_by_step_half_blur와 동일하게 실제 배치 성공 segment 중 정확히 절반에만
        # blur를 적용한다. 홀수개이면 non-blur 쪽에 1개가 더 들어간다.
        n_pasted = len(pending_pastes)
        blur_count = n_pasted // 2
        blur_flags = [True] * blur_count + [False] * (n_pasted - blur_count)
        random.shuffle(blur_flags)

        for item, do_blur in zip(pending_pastes, blur_flags):
            seg_to_paste = (
                apply_segment_shake_blur(item["seg"], prob=1.0)
                if do_blur
                else item["seg"]
            )
            paste_segment(aug, seg_to_paste, item["mask"], item["x"], item["y"])

        if new_lines:
            # 합성 이미지 저장
            syn_img_name = f"syn{syn_counter:05d}.jpg"
            syn_lbl_name = f"syn{syn_counter:05d}.txt"
            cv2.imwrite(str(output_img_dir / syn_img_name), aug)
            with open(output_lbl_dir / syn_lbl_name, "w") as f:
                for l in existing_lines + new_lines:
                    f.write(l + "\n")
            syn_count += 1
            syn_counter += 1
        elif include_failed:
            # needed개 중 단 하나도 세그먼트를 붙이지 못한 경우 -> 원본 그대로 저장
            _save_raw_fallback(img, img_path, lbl_path)

    total_written = syn_count + raw_count
    assert total_written > 0, f"{total_written}개 이미지 생성으로 에러"
    print(f"[synthesis] 합성 완료: synthesized={syn_count}, raw_fallback={raw_count}, total={total_written}")
    return total_written


def _upload_synthesis_to_minio(client, bucket: str, division_idx: str, train_product_idx: str, product_eng_name: str, local_img_dir: Path, local_lbl_dir: Path) -> bool:
    """
    합성 결과를 MinIO divisionSynthesis/{divisionidx}_{trainProductIdx}_{engname}/ 에 업로드.
    """
    prefix = f"divisionSynthesis/{division_idx}_{train_product_idx}_{product_eng_name}"
    
    try:
        image_files = {path.stem: path for path in local_img_dir.glob("*.jpg")}
        label_files = {path.stem: path for path in local_lbl_dir.glob("*.txt")}
        if not image_files or image_files.keys() != label_files.keys():
            missing_images = sorted(label_files.keys() - image_files.keys())
            missing_labels = sorted(image_files.keys() - label_files.keys())
            print(
                f"[synthesis] 업로드 파일 쌍 불일치: "
                f"missing_images={missing_images}, missing_labels={missing_labels}",
                file=sys.stderr,
            )
            return False

        for stem in sorted(image_files):
            for local_file in (image_files[stem], label_files[stem]):
                obj_name = f"{prefix}/{local_file.name}"
                client.fput_object(bucket, obj_name, str(local_file))

        raw_count = sum(stem.startswith("raw") for stem in image_files)
        print(
            f"[synthesis] MinIO 업로드 완료: {bucket}/{prefix} "
            f"(pairs={len(image_files)}, raw_fallback={raw_count})"
        )
        return True
    except Exception as e:
        print(f"[synthesis] MinIO 업로드 실패: {e}", file=sys.stderr)
        return False


def run_synthesis_job(division_idx: str, product_list: list, is_cold: bool = False) -> bool:
    """
    합성 job 실행.
    
    1. MinIO에서 각 product의 pkl 다운로드
    2. Segment bank 구축
    3. 원본 이미지/라벨 다운로드 (MinIO productImg, productAnnotation)
    4. 합성 실행
    5. 결과를 MinIO divisionSynthesis/ 에 업로드
    
    Args:
        division_idx: division index
        product_list: product 정보 리스트
        is_cold: 냉동 여부
    
    Returns:
        성공 여부
    """
    import tempfile
    import shutil
    
    print(f"[synthesis] 합성 job 시작: division_idx={division_idx}, product_count={len(product_list)}")
    
    if not product_list:
        print("[synthesis] product_list가 비어있습니다")
        return False
    
    client = _create_minio_client()
    if not client:
        return False
    
    bucket = os.getenv("MINIO_BUCKET", "chaiimage").strip()
    
    # 임시 작업 디렉토리 생성
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"synthesis_{division_idx}_"))
    pkl_dir = tmp_dir / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. segment bank 로드
        segment_bank = _load_segment_bank(product_list, pkl_dir, client, bucket)
        if not segment_bank['top'] and not segment_bank['side']:
            print("[synthesis] segment bank가 비어있습니다")
            return False
        
        print(f"[synthesis] segment bank 로드 완료: top={len(segment_bank['top'])} classes, side={len(segment_bank['side'])} classes")
        
        # 2. 각 product별 원본 이미지/라벨 다운로드 및 합성
        for product in product_list:
            train_product_idx = str(product.get("train_product_idx") or "").strip()
            product_eng_name = (product.get("product_eng_name") or "").strip()
            foldername = (product.get("foldername") or "").strip()
            
            if not train_product_idx or not product_eng_name:
                print(f"[synthesis] train_product_idx 또는 product_eng_name 누락: {product}")
                continue
            
            if not foldername:
                foldername = f"{train_product_idx}_{product_eng_name}"
            
            print(f"[synthesis] 처리 중: {foldername}")
            
            # 원본 이미지/라벨 다운로드
            product_img_dir = tmp_dir / "original" / foldername / "images"
            product_lbl_dir = tmp_dir / "original" / foldername / "labels"
            product_img_dir.mkdir(parents=True, exist_ok=True)
            product_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            img_prefix = f"productImg/{foldername}"
            lbl_prefix = f"productAnnotation/{foldername}"
            
            # 이미지 다운로드
            img_count = 0
            try:
                for obj in client.list_objects(bucket, prefix=img_prefix + "/", recursive=True):
                    name_lower = obj.object_name.lower()
                    if any(name_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
                        rel_path = obj.object_name[len(img_prefix):].lstrip("/")
                        dest = product_img_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        client.fget_object(bucket, obj.object_name, str(dest))
                        img_count += 1
            except Exception as e:
                print(f"[synthesis] 이미지 다운로드 오류: {e}")
            
            # 라벨 다운로드
            #
            # foldername 끝의 타임스탬프가 단계마다 다르다.
            #   이미지 폴더  = 촬영일  (…_20260731_163750)
            #   검수 라벨    = 검수일  (…_20260804_105250)
            # 그래서 productAnnotation/{foldername} 고정 경로로는 못 찾는다.
            # segment_job.py 와 동일하게 {train_product_idx}_{eng_name}_ 로 탐색한 뒤
            # 가장 최근(사전순 최대) 폴더를 쓴다.
            lbl_count = 0
            lbl_prefix_used = _resolve_annotation_prefix(
                client, bucket, foldername, train_product_idx, product_eng_name
            )
            if not lbl_prefix_used:
                print(
                    f"[synthesis] 검수 라벨 폴더를 찾지 못했습니다: "
                    f"productAnnotation/{train_product_idx}_{product_eng_name}_*",
                    file=sys.stderr,
                )
            else:
                if lbl_prefix_used != lbl_prefix:
                    print(f"[synthesis] 라벨 폴더 탐색: {lbl_prefix_used}")
                try:
                    for obj in client.list_objects(bucket, prefix=lbl_prefix_used + "/", recursive=True):
                        if obj.object_name.lower().endswith(".txt"):
                            rel_path = obj.object_name[len(lbl_prefix_used):].lstrip("/")
                            dest = product_lbl_dir / rel_path
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            client.fget_object(bucket, obj.object_name, str(dest))
                            lbl_count += 1
                except Exception as e:
                    print(f"[synthesis] 라벨 다운로드 오류: {e}")
            
            print(f"[synthesis] 원본 데이터 다운로드: images={img_count}, labels={lbl_count}")
            
            if img_count == 0:
                print(f"[synthesis] 이미지가 없어 건너뜀: {foldername}")
                continue
            
            # 이미지/라벨 경로 수집
            image_paths = list(product_img_dir.rglob("*.jpg")) + list(product_img_dir.rglob("*.png"))
            label_paths = list(product_lbl_dir.rglob("*.txt"))
            
            # 합성 실행
            syn_img_dir = tmp_dir / "synthesis" / foldername / "images"
            syn_lbl_dir = tmp_dir / "synthesis" / foldername / "labels"
            
            # include_failed=True 이면 합성 조건을 만족하지 못한 이미지도 원본 그대로
            # 저장한다. 라벨이 없으면 빈 라벨(negative sample)이 되어 배경 오탐 억제에
            # 기여한다. 레퍼런스 구현(run_training_combined.py)도 라벨 없는 이미지를
            # background 로 학습에 포함한다.
            include_failed = os.getenv("SYNTH_INCLUDE_FAILED", "true").strip().lower() == "true"
            step_by_step = os.getenv("SYNTH_STEP_BY_STEP", "true").strip().lower() == "true"
            augment_ratio = float(os.getenv("SYNTH_AUGMENT_RATIO", "0.5"))
            syn_count = synthesize_on_images(
                segment_bank=segment_bank,
                image_paths=image_paths,
                label_paths=label_paths,
                output_img_dir=syn_img_dir,
                output_lbl_dir=syn_lbl_dir,
                img_root=product_img_dir,
                lbl_root=product_lbl_dir,
                augment_ratio=augment_ratio,
                step_by_step=step_by_step,
                include_failed=include_failed,
            )
            
            if syn_count > 0:
                # MinIO 업로드
                _upload_synthesis_to_minio(client, bucket, division_idx, train_product_idx, product_eng_name, syn_img_dir, syn_lbl_dir)
        
        print(f"[synthesis] 합성 job 완료")
        return True
    
    except Exception as e:
        print(f"[synthesis] 합성 job 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 임시 디렉토리 삭제
        if os.getenv("SYNTHESIS_KEEP_TMP", "false").strip().lower() != "true":
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"[synthesis] 임시 데이터 유지: {tmp_dir}")