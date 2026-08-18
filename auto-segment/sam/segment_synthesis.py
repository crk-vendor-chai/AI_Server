#!/usr/bin/env python3
"""
Segment 합성 스크립트

이미지와 라벨을 받아서 SAM을 이용해 segment 추출 후 PKL 파일을 생성하고 MinIO에 업로드합니다.
- cam_0: 상단 이미지와 라벨 사용
- cam_2: 측면 이미지와 라벨 사용 (선택적)

출력:
- segments/: 합성된 segment 결과
- masks/: 마스크 결과
- visualized/: 시각화 결과
- *.pkl: 최종 합성 결과 (train_product_idx_engname_날짜.pkl 형식)
"""

# SEGMENT_SHARED_PROGRESS_V1
import sys as _segment_progress_sys
import threading as _segment_progress_threading
from pathlib import Path as _SegmentProgressPath


class _SegmentProgressTee:
    def __init__(self, primary, sink, lock):
        self._primary = primary
        self._sink = sink
        self._lock = lock

    def write(self, data):
        if not data:
            return 0

        with self._lock:
            self._primary.write(data)
            self._primary.flush()

            self._sink.write(data)
            self._sink.flush()

        return len(data)

    def flush(self):
        with self._lock:
            self._primary.flush()
            self._sink.flush()

    def isatty(self):
        try:
            return self._primary.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._primary, "errors", "replace")

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _install_segment_shared_progress():
    work_dir = None
    detected_argument = None

    # 인자 이름에 의존하지 않고, argv에 포함된
    # /segment_work/<상품 폴더>/ 경로를 직접 찾는다.
    for raw_argument in _segment_progress_sys.argv[1:]:
        normalized = str(raw_argument).replace("\\", "/")

        if "/segment_work/" in normalized:
            prefix, remainder = normalized.split(
                "/segment_work/",
                1,
            )

            product_folder = remainder.split("/", 1)[0]

            if not product_folder:
                continue

            work_dir = _SegmentProgressPath(
                prefix
                + "/segment_work/"
                + product_folder
            )

            detected_argument = normalized
            break

        if normalized.startswith("segment_work/"):
            remainder = normalized[len("segment_work/"):]
            product_folder = remainder.split("/", 1)[0]

            if not product_folder:
                continue

            work_dir = (
                _SegmentProgressPath.cwd()
                / "segment_work"
                / product_folder
            )

            detected_argument = normalized
            break

    if work_dir is None:
        print(
            "SEGMENT_PROGRESS_SETUP_SKIPPED: "
            "segment_work 경로를 argv에서 찾지 못함",
            file=_segment_progress_sys.__stderr__,
            flush=True,
        )
        print(
            "SEGMENT_ARGV="
            + repr(_segment_progress_sys.argv),
            file=_segment_progress_sys.__stderr__,
            flush=True,
        )
        return

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress_path = (
        work_dir / "segment_progress.log"
    )

    sink = progress_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    )

    lock = _segment_progress_threading.RLock()

    original_stdout = _segment_progress_sys.stdout
    original_stderr = _segment_progress_sys.stderr

    _segment_progress_sys.stdout = _SegmentProgressTee(
        original_stdout,
        sink,
        lock,
    )

    _segment_progress_sys.stderr = _SegmentProgressTee(
        original_stderr,
        sink,
        lock,
    )

    globals()["_SEGMENT_PROGRESS_SINK"] = sink

    print(
        f"SEGMENT_PROGRESS_LOG={progress_path}",
        flush=True,
    )
    print(
        f"SEGMENT_PROGRESS_ARG={detected_argument}",
        flush=True,
    )


_install_segment_shared_progress()


import argparse
import cv2
import numpy as np
import os
import pickle
import re
import sys
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    print("[warning] minio 모듈을 찾을 수 없습니다. MinIO 업로드 기능이 비활성화됩니다.", file=sys.stderr)

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[error] segment_anything 모듈을 찾을 수 없습니다.", file=sys.stderr)
    sys.exit(1)

# =======================
# ⚙️ SAM 설정
# =======================
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "./sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =======================
# ⚙️ 품질 필터링 설정
# =======================
IMG_SIZE = int(os.getenv("SEGMENT_IMG_SIZE", "480"))
MIN_BBOX_AREA = int(os.getenv("MIN_BBOX_AREA", "2000"))
MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE", "2000.0"))
MIN_SAM_CONFIDENCE = float(os.getenv("MIN_SAM_CONFIDENCE", "0.95"))
MIN_PURITY = float(os.getenv("MIN_PURITY", "0.25"))
SEGMENTS_PER_CLASS = int(os.getenv("SEGMENTS_PER_CLASS", "30"))  # 클래스당 추출할 segment 개수

W_CONF = float(os.getenv("SEGMENT_W_CONF", "0.90"))
W_PURITY = float(os.getenv("SEGMENT_W_PURITY", "0.05"))
W_BLUR = float(os.getenv("SEGMENT_W_BLUR", "0.05"))

# =======================
# ⚙️ 적응형 다양성 선발 설정
# =======================
BLUR_PERCENTILE = float(os.getenv("BLUR_PERCENTILE", "85"))
RESERVE_BLUR_PERCENTILES = os.getenv(
    "RESERVE_BLUR_PERCENTILES",
    "60,40,20,0",
)
DIVERSITY_SIM_THRESHOLD = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.92"))
DIVERSITY_SIM_SPREAD_THRESHOLD = float(
    os.getenv("DIVERSITY_SIM_SPREAD_THRESHOLD", "0.02")
)
DIVERSITY_SATURATION_PATIENCE = max(
    1,
    int(os.getenv("DIVERSITY_SATURATION_PATIENCE", "2")),
)
DIVERSITY_NOVEL_TOP_N = max(
    1,
    int(os.getenv("DIVERSITY_NOVEL_TOP_N", "10")),
)
DIVERSITY_POOL_MULTIPLIER = max(
    1,
    int(os.getenv("DIVERSITY_POOL_MULTIPLIER", "10")),
)
DIVERSITY_MIN_POOL_SIZE = max(
    1,
    int(os.getenv("DIVERSITY_MIN_POOL_SIZE", "50")),
)
SIMILARITY_CANVAS_SIZE = max(
    32,
    int(os.getenv("SIMILARITY_CANVAS_SIZE", "100")),
)

# =======================
# 🛠️ 유틸 함수
# =======================

def find_image_file(img_dir, base_name):
    """이미지 파일 찾기 (여러 확장자 지원)"""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = os.path.join(img_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return None

def yolo_to_xyxy(x, y, w, h, img_size):
    """YOLO 형식 (normalized center, width, height)을 xyxy 형식으로 변환"""
    xc, yc, bw, bh = x * img_size, y * img_size, w * img_size, h * img_size
    return [
        max(0, int(xc - bw / 2)),
        max(0, int(yc - bh / 2)),
        min(img_size, int(xc + bw / 2)),
        min(img_size, int(yc + bh / 2)),
    ]

def get_blur_score(img_crop, mask_crop):
    """이미지 블러 점수 계산 (Laplacian variance)"""
    if img_crop is None or img_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = lap[mask_crop > 0]
    return lap_vals.var() if lap_vals.size > 0 else 0.0

def _prepare_similarity_input(candidate, canvas_size=SIMILARITY_CANVAS_SIZE):
    """SAM mask 내부 상품만 비율 유지 resize + padding하여 유사도 입력을 생성합니다."""
    segment = candidate.get("segment")
    mask = candidate.get("mask")

    if segment is None or segment.size == 0:
        return None, None

    if mask is None or mask.size == 0:
        mask = np.ones(segment.shape[:2], dtype=np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    points = cv2.findNonZero(mask)
    if points is None:
        return None, None

    x, y, w, h = cv2.boundingRect(points)
    if w <= 0 or h <= 0:
        return None, None

    object_crop = segment[y:y + h, x:x + w]
    mask_crop = mask[y:y + h, x:x + w]
    if object_crop.size == 0 or mask_crop.size == 0:
        return None, None

    scale = min(canvas_size / float(w), canvas_size / float(h))
    resized_w = max(1, min(canvas_size, int(round(w * scale))))
    resized_h = max(1, min(canvas_size, int(round(h * scale))))

    object_resized = cv2.resize(
        object_crop,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    mask_resized = cv2.resize(
        mask_crop,
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    mask_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x_offset = (canvas_size - resized_w) // 2
    y_offset = (canvas_size - resized_h) // 2

    canvas[
        y_offset:y_offset + resized_h,
        x_offset:x_offset + resized_w,
    ] = object_resized
    mask_canvas[
        y_offset:y_offset + resized_h,
        x_offset:x_offset + resized_w,
    ] = mask_resized

    canvas = cv2.bitwise_and(canvas, canvas, mask=mask_canvas)
    return canvas, mask_canvas


def _get_similarity_hist(candidate):
    """후보별 mask-aware HSV histogram을 한 번만 계산하여 캐시합니다."""
    cached = candidate.get("_similarity_hist")
    if cached is not None:
        return cached

    image, mask = _prepare_similarity_input(candidate)
    if image is None or mask is None or not np.any(mask):
        hist = np.zeros((8, 8, 8), dtype=np.float32)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            mask,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)

    candidate["_similarity_hist"] = hist
    return hist


def compute_similarity(a, b):
    """SAM mask 내부 상품의 HSV histogram correlation을 계산합니다."""
    h1 = _get_similarity_hist(a)
    h2 = _get_similarity_hist(b)

    if not np.any(h1) or not np.any(h2):
        return 1.0

    similarity = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    if not np.isfinite(similarity):
        return 1.0
    return similarity

def save_combined_preview(candidates, train_product_idx, product_eng_name, cam_name, visualized_dir):
    """선택된 segment들을 격자형으로 만들어 previews에 저장"""
    if not candidates:
        return None
    
    tile_size = 150
    cols = 10
    rows = (len(candidates) + cols - 1) // cols
    canvas = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    
    for idx, cand in enumerate(candidates):
        seg_masked = cv2.bitwise_and(cand['segment'], cand['segment'], mask=cand['mask'])
        seg_res = cv2.resize(seg_masked, (tile_size, tile_size))
        r, c = divmod(idx, cols)
        canvas[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = seg_res
    
    # 상품별 디렉토리 생성
    product_dir = os.path.join(visualized_dir, f"{train_product_idx}_{product_eng_name}")
    os.makedirs(product_dir, exist_ok=True)
    
    preview_path = os.path.join(product_dir, f"{cam_name}_preview.jpg")
    cv2.imwrite(preview_path, canvas)
    return preview_path

# =======================
# 🎯 다양성 선발 함수
# =======================

def _set_quality_score(candidate):
    """환경변수 가중치를 사용해 모든 후보에 동일한 품질 점수를 부여합니다."""
    norm_blur = min(max(float(candidate["blur"]) / 5000.0, 0.0), 1.0)
    weight_sum = W_CONF + W_PURITY + W_BLUR
    if weight_sum <= 0:
        weight_sum = 1.0

    score = (
        W_CONF * float(candidate["conf"])
        + W_PURITY * float(candidate["purity"])
        + W_BLUR * norm_blur
    ) / weight_sum

    candidate["total_score"] = score
    candidate["q_score"] = score
    return score


def _remove_candidate_by_identity(pool, target):
    """numpy 배열을 포함한 dict도 안전하게 identity 기준으로 제거합니다."""
    for index, candidate in enumerate(pool):
        if candidate is target:
            pool.pop(index)
            return True
    return False


def diverse_selection(candidates, k):
    """기존 호출과의 호환성을 위한 mask-aware 다양성 선발 함수입니다."""
    if not candidates or k <= 0:
        return []

    pool = list(candidates)
    for candidate in pool:
        _set_quality_score(candidate)
    pool.sort(key=lambda item: item["q_score"], reverse=True)

    selected = [pool.pop(0)]
    selected[0]["selection_source"] = selected[0].get("pool_name", "fallback")
    selected[0]["selection_max_similarity"] = None

    while len(selected) < k and pool:
        scored = []
        for candidate in pool:
            max_similarity = max(
                compute_similarity(candidate, chosen)
                for chosen in selected
            )
            scored.append((candidate, max_similarity))

        scored.sort(key=lambda item: item[1])
        novel_candidates = scored[:min(DIVERSITY_NOVEL_TOP_N, len(scored))]
        chosen, chosen_similarity = max(
            novel_candidates,
            key=lambda item: item[0]["q_score"],
        )
        chosen["selection_source"] = chosen.get("pool_name", "fallback")
        chosen["selection_max_similarity"] = float(chosen_similarity)
        selected.append(chosen)
        _remove_candidate_by_identity(pool, chosen)

    return selected


def build_quality_diverse_pool(candidates, pool_size):
    """품질 순위의 상·중·하 구간에서 후보를 균등 샘플링합니다."""
    if not candidates or pool_size <= 0:
        return []
    if len(candidates) <= pool_size:
        return list(candidates)

    ordered = sorted(candidates, key=lambda item: item["q_score"], reverse=True)
    n = len(ordered)

    high = ordered[:n // 3]
    middle = ordered[n // 3:2 * n // 3]
    low = ordered[2 * n // 3:]

    high_count = int(pool_size * 0.50)
    middle_count = int(pool_size * 0.35)
    low_count = pool_size - high_count - middle_count

    def _sample(group, count):
        if not group or count <= 0:
            return []
        indices = np.linspace(
            0,
            len(group) - 1,
            min(count, len(group)),
            dtype=int,
        )
        return [group[index] for index in indices]

    sampled = (
        _sample(high, high_count)
        + _sample(middle, middle_count)
        + _sample(low, low_count)
    )

    # 매우 작은 구간 때문에 목표 수를 채우지 못한 경우 남은 품질 순 후보로 보충합니다.
    sampled_ids = {id(candidate) for candidate in sampled}
    if len(sampled) < pool_size:
        for candidate in ordered:
            if id(candidate) in sampled_ids:
                continue
            sampled.append(candidate)
            sampled_ids.add(id(candidate))
            if len(sampled) >= pool_size:
                break

    return sampled


def _limit_selection_pool(candidates, k):
    """유사도 계산량을 제한하면서 품질 구간의 다양성은 유지합니다."""
    if not candidates:
        return []

    pool_size = min(
        len(candidates),
        max(k * DIVERSITY_POOL_MULTIPLIER, DIVERSITY_MIN_POOL_SIZE),
    )
    return build_quality_diverse_pool(candidates, pool_size)


def _parse_reserve_percentiles(main_percentile):
    """reserve 경계 percentile을 내림차순으로 정리합니다."""
    values = []
    for raw_value in RESERVE_BLUR_PERCENTILES.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            print(
                f"[warning] 잘못된 RESERVE_BLUR_PERCENTILES 값 무시: {raw_value}",
                file=sys.stderr,
            )
            continue
        value = min(max(value, 0.0), 100.0)
        if value < main_percentile:
            values.append(value)

    # 마지막 데이터까지 fallback으로 사용할 수 있도록 0 percentile을 보장합니다.
    values.append(0.0)
    return sorted(set(values), reverse=True)


def build_blur_pools(candidates, main_percentile=BLUR_PERCENTILE):
    """후보를 main과 단계별 reserve blur band로 분리합니다."""
    if not candidates:
        return [], [], {
            "main_percentile": float(main_percentile),
            "main_threshold": MIN_BLUR_SCORE,
            "reserve_percentiles": [],
            "reserve_thresholds": [],
        }

    blur_values = np.asarray(
        [float(candidate["blur"]) for candidate in candidates],
        dtype=np.float64,
    )
    main_percentile = min(max(float(main_percentile), 0.0), 100.0)
    main_threshold = float(np.percentile(blur_values, main_percentile))

    reserve_percentiles = _parse_reserve_percentiles(main_percentile)
    reserve_thresholds = [
        float(np.percentile(blur_values, percentile))
        for percentile in reserve_percentiles
    ]

    main_pool = []
    reserve_bands = [[] for _ in reserve_percentiles]

    for candidate in candidates:
        blur = float(candidate["blur"])
        if blur >= main_threshold:
            candidate["pool_name"] = f"main_p{main_percentile:g}+"
            main_pool.append(candidate)
            continue

        assigned = False
        upper_percentile = main_percentile
        for index, (lower_percentile, lower_threshold) in enumerate(
            zip(reserve_percentiles, reserve_thresholds)
        ):
            if blur >= lower_threshold:
                candidate["pool_name"] = (
                    f"reserve_p{lower_percentile:g}_to_p{upper_percentile:g}"
                )
                reserve_bands[index].append(candidate)
                assigned = True
                break
            upper_percentile = lower_percentile

        if not assigned:
            # 수치 오차나 음수 score가 있더라도 최종 reserve에서 보존합니다.
            if not reserve_bands:
                candidate["pool_name"] = "reserve_all"
                reserve_bands.append([candidate])
                reserve_percentiles.append(0.0)
                reserve_thresholds.append(float(blur_values.min()))
            else:
                candidate["pool_name"] = "reserve_below_p0"
                reserve_bands[-1].append(candidate)

    metadata = {
        "main_percentile": main_percentile,
        "main_threshold": main_threshold,
        "reserve_percentiles": reserve_percentiles,
        "reserve_thresholds": reserve_thresholds,
    }
    return main_pool, reserve_bands, metadata


def adaptive_diverse_selection(
    main_candidates,
    reserve_bands,
    k,
    log_prefix="",
):
    """
    Main pool에서 선발하고, 유사도 포화 또는 후보 부족 시 reserve band를
    선명도 순으로 단계적으로 개방합니다.
    """
    if k <= 0:
        return [], {
            "reserve_bands_opened": 0,
            "saturation_events": 0,
            "iterations": [],
        }

    all_candidates = list(main_candidates)
    for band in reserve_bands:
        all_candidates.extend(band)
    if not all_candidates:
        return [], {
            "reserve_bands_opened": 0,
            "saturation_events": 0,
            "iterations": [],
        }

    for candidate in all_candidates:
        _set_quality_score(candidate)

    main_pool = _limit_selection_pool(main_candidates, k)
    reserve_pools = [_limit_selection_pool(band, k) for band in reserve_bands]

    active_pool = list(main_pool)
    next_reserve_index = 0
    reserve_bands_opened = 0
    saturation_count = 0
    saturation_events = 0
    iteration_logs = []

    def _open_next_reserve(reason):
        nonlocal next_reserve_index, reserve_bands_opened, saturation_count

        while next_reserve_index < len(reserve_pools):
            band = reserve_pools[next_reserve_index]
            next_reserve_index += 1
            if not band:
                continue

            existing_ids = {id(candidate) for candidate in active_pool}
            added = [candidate for candidate in band if id(candidate) not in existing_ids]
            active_pool.extend(added)
            reserve_bands_opened += 1
            saturation_count = 0

            pool_name = added[0].get("pool_name", f"reserve_{next_reserve_index}") if added else "empty"
            print(
                f"{log_prefix}reserve open: reason={reason}, "
                f"band={pool_name}, added={len(added)}, active={len(active_pool)}"
            )
            return bool(added)

        return False

    if not active_pool:
        _open_next_reserve("main_empty")
    if not active_pool:
        return [], {
            "reserve_bands_opened": reserve_bands_opened,
            "saturation_events": saturation_events,
            "iterations": iteration_logs,
        }

    active_pool.sort(key=lambda item: item["q_score"], reverse=True)
    first = active_pool.pop(0)
    first["selection_source"] = first.get("pool_name", "main")
    first["selection_max_similarity"] = None
    first["selection_iteration"] = 0
    selected = [first]

    while len(selected) < k:
        if not active_pool:
            if _open_next_reserve("active_pool_empty"):
                continue
            break

        similarity_scores = []
        for candidate in active_pool:
            max_similarity = max(
                compute_similarity(candidate, chosen)
                for chosen in selected
            )
            similarity_scores.append((candidate, float(max_similarity)))

        similarities = np.asarray(
            [similarity for _, similarity in similarity_scores],
            dtype=np.float64,
        )
        best_similarity = float(similarities.min())
        worst_similarity = float(similarities.max())
        spread = worst_similarity - best_similarity
        saturated = (
            best_similarity >= DIVERSITY_SIM_THRESHOLD
            and spread <= DIVERSITY_SIM_SPREAD_THRESHOLD
        )

        iteration_log = {
            "selected_count_before": len(selected),
            "active_pool_count": len(active_pool),
            "best_max_similarity": best_similarity,
            "worst_max_similarity": worst_similarity,
            "similarity_spread": spread,
            "saturated": bool(saturated),
        }
        iteration_logs.append(iteration_log)

        print(
            f"{log_prefix}selection={len(selected):02d}/{k}, "
            f"active={len(active_pool)}, "
            f"best_sim={best_similarity:.4f}, "
            f"worst_sim={worst_similarity:.4f}, "
            f"spread={spread:.4f}, saturated={saturated}"
        )

        if saturated:
            saturation_count += 1
        else:
            saturation_count = 0

        if (
            saturation_count >= DIVERSITY_SATURATION_PATIENCE
            and next_reserve_index < len(reserve_pools)
        ):
            saturation_events += 1
            if _open_next_reserve("similarity_saturated"):
                continue

        similarity_scores.sort(key=lambda item: item[1])
        novel_candidates = similarity_scores[
            :min(DIVERSITY_NOVEL_TOP_N, len(similarity_scores))
        ]

        chosen, chosen_similarity = max(
            novel_candidates,
            key=lambda item: item[0]["q_score"],
        )
        chosen["selection_source"] = chosen.get("pool_name", "active")
        chosen["selection_max_similarity"] = float(chosen_similarity)
        chosen["selection_iteration"] = len(selected)
        selected.append(chosen)
        _remove_candidate_by_identity(active_pool, chosen)

    # PKL에 불필요한 histogram cache가 저장되지 않도록 제거합니다.
    for candidate in all_candidates:
        candidate.pop("_similarity_hist", None)

    diagnostics = {
        "reserve_bands_opened": reserve_bands_opened,
        "saturation_events": saturation_events,
        "iterations": iteration_logs,
        "selected_from_main": sum(
            1 for candidate in selected
            if str(candidate.get("selection_source", "")).startswith("main_")
        ),
        "selected_from_reserve": sum(
            1 for candidate in selected
            if str(candidate.get("selection_source", "")).startswith("reserve_")
        ),
    }
    return selected, diagnostics


def diverse_select2(candidates, k):
    """기존 API 호환용: reserve 없이 adaptive selector를 사용합니다."""
    selected, _ = adaptive_diverse_selection(candidates, [], k)
    return selected

# =======================
# 🔍 이미지/라벨 찾기 함수
# =======================

def detect_side_cam(images_dir: str):
    """이미지 디렉터리에서 side 로 쓸 카메라를 찾는다.

    cam_0 은 top 으로 고정이고, 그 외에 존재하는 카메라를 side 로 본다.
    리그마다 side 번호가 달라서(_C=cam_2, _F=cam_1) 하드코딩하지 않는다.
    여러 개면 번호가 가장 작은 것을 쓴다.
    """
    root = Path(images_dir)
    if not root.exists():
        return None

    found = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        matched = re.match(r"^cam[_-]?(\d+)$", entry.name.lower())
        if not matched:
            continue
        num = int(matched.group(1))
        if num == 0:
            continue
        found.setdefault(num, entry.name)

    if not found:
        return None
    return f"cam_{min(found)}"


def find_cam_images_labels(images_dir: str, labels_dir: str, cam_name: str = "cam_0"):
    """
    특정 카메라의 이미지와 라벨을 찾습니다.
    
    Args:
        images_dir: 이미지 디렉토리 (cam_0/, cam_2/ 등 포함)
        labels_dir: 라벨 디렉토리 (cam_0/, cam_2/ 등 포함)
        cam_name: 카메라 이름 (예: "cam_0", "cam_2")
    
    Returns:
        (image_paths, label_paths): 이미지 경로 리스트, 라벨 경로 리스트
    """
    # cam_0 / cam0 처럼 표기가 섞여 있으므로 둘 다 시도한다.
    # 카메라 번호에 관계없이 동작 (cam_0, cam_1, cam_2, ...).
    cam_aliases = [cam_name]
    _cam_match = re.match(r"^cam[_-]?(\d+)$", cam_name.lower())
    if _cam_match:
        _n = int(_cam_match.group(1))
        for _alias in (f"cam_{_n}", f"cam{_n}"):
            if _alias not in cam_aliases:
                cam_aliases.append(_alias)

    image_paths = []
    for alias in cam_aliases:
        images_path = Path(images_dir) / alias
        if not images_path.exists():
            continue
        img_subdir = images_path / "images"
        if img_subdir.exists():
            image_paths = sorted(img_subdir.glob("*.jpg")) + sorted(img_subdir.glob("*.png")) +                          sorted(img_subdir.glob("*.jpeg")) + sorted(img_subdir.glob("*.JPG")) +                          sorted(img_subdir.glob("*.PNG")) + sorted(img_subdir.glob("*.JPEG"))
        else:
            image_paths = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png")) +                          sorted(images_path.glob("*.jpeg")) + sorted(images_path.glob("*.JPG")) +                          sorted(images_path.glob("*.PNG")) + sorted(images_path.glob("*.JPEG"))
        if image_paths:
            break

    label_paths = []
    for alias in cam_aliases:
        labels_path = Path(labels_dir) / alias
        if labels_path.exists():
            label_paths = sorted(labels_path.rglob("*.txt"))
            if label_paths:
                break

    return image_paths, label_paths

def match_image_label(image_paths, label_paths):
    """이미지와 라벨을 매칭"""
    matched = []
    label_dict = {Path(p).stem: p for p in label_paths}
    
    for img_path in image_paths:
        base_name = Path(img_path).stem
        if base_name in label_dict:
            matched.append((img_path, label_dict[base_name]))
    
    return matched

# =======================
# 🚀 Segment 추출 함수
# =======================

# def extract_segments_for_cam(images_dir: str, labels_dir: str, cam_name: str, 
#                              predictor: SamPredictor, out_segment_dir: str, 
#                              out_mask_dir: str, out_visualized_dir: str,
#                              train_product_idx: str, product_eng_name: str):
#     """
#     특정 카메라의 segment를 추출합니다.
    
#     Args:
#         images_dir: 이미지 디렉토리
#         labels_dir: 라벨 디렉토리
#         cam_name: 카메라 이름 ("cam_0" 또는 "cam_2")
#         predictor: SAM predictor
#         out_segment_dir: segment 출력 디렉토리
#         out_mask_dir: mask 출력 디렉토리
#         out_visualized_dir: 시각화 출력 디렉토리
#         train_product_idx: train_product_idx
#         product_eng_name: product_eng_name
    
#     Returns:
#         segment_data: 추출된 segment 데이터 (dict)
#     """
#     print(f"[segment_synthesis] {cam_name} segment 추출 시작")
    
#     image_paths, label_paths = find_cam_images_labels(images_dir, labels_dir, cam_name)
    
#     if not image_paths:
#         print(f"[segment_synthesis] 경고: {cam_name} 이미지가 없습니다", file=sys.stderr)
#         return None
    
#     if not label_paths:
#         print(f"[segment_synthesis] 경고: {cam_name} 라벨이 없습니다", file=sys.stderr)
#         return None
    
#     print(f"[segment_synthesis] {cam_name}: 이미지 {len(image_paths)}개, 라벨 {len(label_paths)}개")
    
#     # 이미지와 라벨 매칭
#     matched = match_image_label(image_paths, label_paths)
#     if not matched:
#         print(f"[segment_synthesis] 경고: {cam_name}에서 매칭된 이미지-라벨 쌍이 없습니다", file=sys.stderr)
#         return None
    
#     print(f"[segment_synthesis] {cam_name}: 매칭된 쌍 {len(matched)}개")
    
#     # 모든 클래스의 segment 후보 수집
#     all_candidates = []

#     # FILTER_STATS_BEGIN
#     filter_stats = {
#         "target_labels": 0,
#         "bbox_reject": 0,
#         "bbox_pass": 0,
#         "sam_conf_reject": 0,
#         "sam_conf_pass": 0,
#         "empty_crop": 0,
#         "purity_reject": 0,
#         "purity_pass": 0,
#         "blur_reject": 0,
#         "blur_pass": 0,
#         "accepted": 0,
#     }

#     sam_conf_values = []
#     purity_values = []
#     blur_values = []
#     # FILTER_STATS_END
    
#     for img_path, lbl_path in tqdm(matched, desc=f"   {cam_name.upper()}", leave=False):
#         img = cv2.imread(str(img_path))
#         if img is None:
#             continue
        
#         # 이미지 크기 조정 (필요시)
#         if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
#         with open(lbl_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 try:
#                     cls_id = int(parts[0])
#                     if cls_id == 0:  # background 클래스 제외
#                         continue
#                     x, y, w, h = map(float, parts[1:5])
#                 except (ValueError, IndexError):
#                     continue

#                 filter_stats["target_labels"] += 1

#                 bbox = yolo_to_xyxy(x, y, w, h, IMG_SIZE)
#                 area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#                 if area < MIN_BBOX_AREA:
#                     filter_stats["bbox_reject"] += 1
#                     continue

#                 filter_stats["bbox_pass"] += 1
                
#                 # SAM으로 segment 추출
#                 predictor.set_image(img)
#                 masks, scores, _ = predictor.predict(box=np.array(bbox), multimask_output=False)
                
#                 sam_conf = float(scores[0])
#                 sam_conf_values.append(sam_conf)

#                 if sam_conf < MIN_SAM_CONFIDENCE:
#                     filter_stats["sam_conf_reject"] += 1
#                     continue

#                 filter_stats["sam_conf_pass"] += 1

#                 mask = (masks[0] * 255).astype(np.uint8)
#                 img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#                 mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
#                 if img_crop.size == 0 or mask_crop.size == 0:
#                     filter_stats["empty_crop"] += 1
#                     continue
                
#                 purity = np.sum(mask_crop > 0) / (area + 1e-6)
#                 blur = get_blur_score(img_crop, mask_crop)
                
#                 purity_values.append(float(purity))
#                 blur_values.append(float(blur))

#                 if purity < MIN_PURITY:
#                     filter_stats["purity_reject"] += 1
#                     continue

#                 filter_stats["purity_pass"] += 1

#                 if blur < MIN_BLUR_SCORE:
#                     filter_stats["blur_reject"] += 1
#                     continue

#                 filter_stats["blur_pass"] += 1
#                 filter_stats["accepted"] += 1

#                 norm_blur = min(blur / 5000.0, 1.0)
#                 total_score = W_CONF * sam_conf + W_PURITY * purity + W_BLUR * norm_blur
                
#                 all_candidates.append({
#                     "segment": img_crop,
#                     "mask": mask_crop,
#                     "conf": sam_conf,
#                     "purity": purity,
#                     "blur": blur,
#                     "total_score": total_score,
#                     "cls_id": cls_id,
#                     "img_path": str(img_path),
#                     "lbl_path": str(lbl_path),
#                 })
    
#     def _filter_value_stats(values):
#         if not values:
#             return "n=0"

#         arr = np.asarray(values, dtype=np.float64)

#         return (
#             f"n={len(arr)} "
#             f"min={arr.min():.4f} "
#             f"p25={np.percentile(arr, 25):.4f} "
#             f"p50={np.percentile(arr, 50):.4f} "
#             f"p75={np.percentile(arr, 75):.4f} "
#             f"p95={np.percentile(arr, 95):.4f} "
#             f"max={arr.max():.4f}"
#         )

#     print(
#         f"[segment_synthesis] {cam_name}: "
#         f"thresholds="
#         f"bbox_area>={MIN_BBOX_AREA}, "
#         f"sam_conf>={MIN_SAM_CONFIDENCE}, "
#         f"purity>={MIN_PURITY}, "
#         f"blur>={MIN_BLUR_SCORE}"
#     )

#     print(
#         f"[segment_synthesis] {cam_name}: "
#         f"filter_stats={filter_stats}"
#     )

#     print(
#         f"[segment_synthesis] {cam_name}: "
#         f"sam_conf_stats={_filter_value_stats(sam_conf_values)}"
#     )

#     print(
#         f"[segment_synthesis] {cam_name}: "
#         f"purity_stats={_filter_value_stats(purity_values)}"
#     )

#     print(
#         f"[segment_synthesis] {cam_name}: "
#         f"blur_stats={_filter_value_stats(blur_values)}"
#     )

#     if not all_candidates:
#         print(f"[segment_synthesis] 경고: {cam_name}에서 추출된 segment가 없습니다", file=sys.stderr)
#         return None
    
#     print(f"[segment_synthesis] {cam_name}: 총 {len(all_candidates)}개 후보 추출")
    
#     # 클래스별로 그룹화
#     candidates_by_class = defaultdict(list)
#     for cand in all_candidates:
#         candidates_by_class[cand['cls_id']].append(cand)
    
#     # 클래스별로 다양성 선발
#     selected_segments = []
#     for cls_id, candidates in candidates_by_class.items():
#         # 1차 선발
#         selected_1 = diverse_select2(candidates, SEGMENTS_PER_CLASS)
#         min_req = int(SEGMENTS_PER_CLASS * 2 / 3)
        
#         # fallback 필요한 경우
#         if len(selected_1) < min_req:
#             print(f"   ⚠️ {cam_name} class {cls_id}: fallback ({len(selected_1)}<{min_req})")
#             selected_ids = set(id(s) for s in selected_1)
#             remaining = [c for c in candidates if id(c) not in selected_ids]
#             fallback = diverse_selection(remaining, SEGMENTS_PER_CLASS)
#             need = SEGMENTS_PER_CLASS - len(selected_1)
#             selected = selected_1 + fallback[:need]
#         else:
#             selected = selected_1
        
#         selected_segments.extend(selected)
    
#     if not selected_segments:
#         print(f"[segment_synthesis] 경고: {cam_name}에서 선발된 segment가 없습니다", file=sys.stderr)
#         return None
    
#     print(f"[segment_synthesis] {cam_name}: {len(selected_segments)}개 segment 선발 완료")
    
#     # 저장
#     cam_out_dir = os.path.join(out_segment_dir, cam_name)
#     cam_mask_dir = os.path.join(out_mask_dir, cam_name)
#     os.makedirs(cam_out_dir, exist_ok=True)
#     os.makedirs(cam_mask_dir, exist_ok=True)
    
#     for i, s in enumerate(selected_segments):
#         fname = f"{i:02d}_c{s['conf']:.2f}_p{s['purity']:.2f}_b{int(s['blur'])}.png"
#         seg_bgra = cv2.merge([s["segment"], s["mask"]])
#         cv2.imwrite(os.path.join(cam_out_dir, fname), seg_bgra)
#         cv2.imwrite(os.path.join(cam_mask_dir, fname), s["mask"])
    
#     # PKL 저장
#     pkl_path = os.path.join(cam_out_dir, "segments.pkl")
#     with open(pkl_path, "wb") as f:
#         pickle.dump(selected_segments, f)
    
#     # 시각화 이미지 저장
#     preview_path = save_combined_preview(
#         selected_segments, train_product_idx, product_eng_name, cam_name, out_visualized_dir
#     )
    
#     # 결과 데이터 구성
#     segment_data = {
#         "cam": cam_name,
#         "view": "top" if cam_name == "cam_0" else "side",
#         "image_count": len(image_paths),
#         "label_count": len(label_paths),
#         "matched_count": len(matched),
#         "candidates_count": len(all_candidates),
#         "selected_count": len(selected_segments),
#         "segments": selected_segments,
#         "pkl_path": pkl_path,
#         "preview_path": preview_path,
#     }
    
#     print(f"[segment_synthesis] {cam_name} segment 추출 완료")
#     return segment_data

def extract_segments_for_cam(images_dir: str, labels_dir: str, cam_name: str,
                             predictor: SamPredictor, out_segment_dir: str,
                             out_mask_dir: str, out_visualized_dir: str,
                             train_product_idx: str, product_eng_name: str):
    """특정 카메라에서 segment 후보를 만들고 적응형 다양성 선발을 수행합니다."""
    print(f"[segment_synthesis] {cam_name} segment 추출 시작")

    image_paths, label_paths = find_cam_images_labels(images_dir, labels_dir, cam_name)

    if not image_paths:
        print(f"[segment_synthesis] 경고: {cam_name} 이미지가 없습니다", file=sys.stderr)
        return None

    if not label_paths:
        print(f"[segment_synthesis] 경고: {cam_name} 라벨이 없습니다", file=sys.stderr)
        return None

    print(f"[segment_synthesis] {cam_name}: 이미지 {len(image_paths)}개, 라벨 {len(label_paths)}개")

    matched = match_image_label(image_paths, label_paths)
    if not matched:
        print(f"[segment_synthesis] 경고: {cam_name}에서 매칭된 이미지-라벨 쌍이 없습니다", file=sys.stderr)
        return None

    print(f"[segment_synthesis] {cam_name}: 매칭된 쌍 {len(matched)}개")

    # bbox, SAM confidence, purity를 통과한 전체 후보입니다.
    # blur는 hard reject하지 않고 main/reserve pool 분리에만 사용합니다.
    purity_passed = []

    filter_stats = {
        "target_labels": 0,
        "bbox_reject": 0,
        "bbox_pass": 0,
        "sam_conf_reject": 0,
        "sam_conf_pass": 0,
        "empty_crop": 0,
        "purity_reject": 0,
        "purity_pass": 0,
        # blur는 더 이상 hard reject하지 않습니다.
        "blur_reject": 0,
        "blur_pass": 0,
        "blur_reserve": 0,
        "main_candidates": 0,
        "reserve_candidates": 0,
        "accepted": 0,
    }

    sam_conf_values = []
    purity_values = []
    blur_values = []

    for img_path, lbl_path in tqdm(matched, desc=f"   {cam_name.upper()}", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        parsed_labels = []
        with open(lbl_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        continue
                    x, y, w, h = map(float, parts[1:5])
                except (ValueError, IndexError):
                    continue

                filter_stats["target_labels"] += 1
                bbox = yolo_to_xyxy(x, y, w, h, IMG_SIZE)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < MIN_BBOX_AREA:
                    filter_stats["bbox_reject"] += 1
                    continue

                filter_stats["bbox_pass"] += 1
                parsed_labels.append((cls_id, bbox, area))

        if not parsed_labels:
            continue

        # SAM image encoder는 이미지당 한 번만 실행하고, SAM에는 RGB를 전달합니다.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        for cls_id, bbox, area in parsed_labels:
            masks, scores, _ = predictor.predict(
                box=np.asarray(bbox, dtype=np.float32),
                multimask_output=False,
            )

            sam_conf = float(scores[0])
            sam_conf_values.append(sam_conf)
            if sam_conf < MIN_SAM_CONFIDENCE:
                filter_stats["sam_conf_reject"] += 1
                continue

            filter_stats["sam_conf_pass"] += 1

            mask = (masks[0] * 255).astype(np.uint8)
            img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            if img_crop.size == 0 or mask_crop.size == 0:
                filter_stats["empty_crop"] += 1
                continue

            purity = float(np.sum(mask_crop > 0) / (area + 1e-6))
            purity_values.append(purity)
            if purity < MIN_PURITY:
                filter_stats["purity_reject"] += 1
                continue

            filter_stats["purity_pass"] += 1
            blur = float(get_blur_score(img_crop, mask_crop))
            blur_values.append(blur)

            candidate = {
                "segment": img_crop,
                "mask": mask_crop,
                "conf": sam_conf,
                "purity": purity,
                "blur": blur,
                "cls_id": cls_id,
                "img_path": str(img_path),
                "lbl_path": str(lbl_path),
            }
            _set_quality_score(candidate)
            purity_passed.append(candidate)

    def _filter_value_stats(values):
        if not values:
            return "n=0"

        arr = np.asarray(values, dtype=np.float64)
        return (
            f"n={len(arr)} "
            f"min={arr.min():.4f} "
            f"p25={np.percentile(arr, 25):.4f} "
            f"p50={np.percentile(arr, 50):.4f} "
            f"p75={np.percentile(arr, 75):.4f} "
            f"p95={np.percentile(arr, 95):.4f} "
            f"max={arr.max():.4f}"
        )

    if not purity_passed:
        print(
            f"[segment_synthesis] 경고: {cam_name}에서 SAM confidence와 purity를 "
            "통과한 segment가 없습니다",
            file=sys.stderr,
        )
        return None

    candidates_by_class = defaultdict(list)
    for candidate in purity_passed:
        candidates_by_class[candidate["cls_id"]].append(candidate)

    selected_segments = []
    class_selection_diagnostics = {}
    class_pool_metadata = {}

    for cls_id, class_candidates in candidates_by_class.items():
        main_pool, reserve_bands, pool_metadata = build_blur_pools(
            class_candidates,
            BLUR_PERCENTILE,
        )

        main_count = len(main_pool)
        reserve_count = sum(len(band) for band in reserve_bands)
        filter_stats["main_candidates"] += main_count
        filter_stats["reserve_candidates"] += reserve_count
        filter_stats["blur_pass"] += main_count
        filter_stats["blur_reserve"] += reserve_count
        filter_stats["accepted"] += len(class_candidates)

        print(
            f"[segment_synthesis] {cam_name} class {cls_id}: "
            f"main={main_count}, reserve={[len(band) for band in reserve_bands]}, "
            f"main_blur_threshold={pool_metadata['main_threshold']:.4f} "
            f"(p{pool_metadata['main_percentile']:g})"
        )

        selected, diagnostics = adaptive_diverse_selection(
            main_pool,
            reserve_bands,
            SEGMENTS_PER_CLASS,
            log_prefix=f"   [{cam_name} class {cls_id}] ",
        )

        if len(selected) < SEGMENTS_PER_CLASS:
            print(
                f"   ⚠️ {cam_name} class {cls_id}: "
                f"요청={SEGMENTS_PER_CLASS}, 실제 선발={len(selected)}, "
                f"전체 사용 가능 후보={len(class_candidates)}"
            )

        selected_segments.extend(selected)
        class_selection_diagnostics[str(cls_id)] = diagnostics
        class_pool_metadata[str(cls_id)] = {
            **pool_metadata,
            "main_count": main_count,
            "reserve_counts": [len(band) for band in reserve_bands],
        }

    print(
        f"[segment_synthesis] {cam_name}: thresholds="
        f"bbox_area>={MIN_BBOX_AREA}, "
        f"sam_conf>={MIN_SAM_CONFIDENCE}, "
        f"purity>={MIN_PURITY}, "
        f"main_blur_percentile=p{BLUR_PERCENTILE:g}, "
        f"reserve_percentiles={_parse_reserve_percentiles(BLUR_PERCENTILE)}"
    )
    print(f"[segment_synthesis] {cam_name}: filter_stats={filter_stats}")
    print(f"[segment_synthesis] {cam_name}: sam_conf_stats={_filter_value_stats(sam_conf_values)}")
    print(f"[segment_synthesis] {cam_name}: purity_stats={_filter_value_stats(purity_values)}")
    print(f"[segment_synthesis] {cam_name}: blur_stats={_filter_value_stats(blur_values)}")

    if not selected_segments:
        print(f"[segment_synthesis] 경고: {cam_name}에서 선발된 segment가 없습니다", file=sys.stderr)
        return None

    print(f"[segment_synthesis] {cam_name}: {len(selected_segments)}개 segment 선발 완료")

    cam_out_dir = os.path.join(out_segment_dir, cam_name)
    cam_mask_dir = os.path.join(out_mask_dir, cam_name)
    os.makedirs(cam_out_dir, exist_ok=True)
    os.makedirs(cam_mask_dir, exist_ok=True)

    for index, segment in enumerate(selected_segments):
        segment.pop("_similarity_hist", None)
        # 기존 파일명 형식을 유지하여 후속 스크립트와의 호환성을 보장합니다.
        filename = (
            f"{index:02d}_c{segment['conf']:.2f}_p{segment['purity']:.2f}_"
            f"b{int(segment['blur'])}.png"
        )
        segment_bgra = cv2.merge([segment["segment"], segment["mask"]])
        cv2.imwrite(os.path.join(cam_out_dir, filename), segment_bgra)
        cv2.imwrite(os.path.join(cam_mask_dir, filename), segment["mask"])

    pkl_path = os.path.join(cam_out_dir, "segments.pkl")
    with open(pkl_path, "wb") as file:
        pickle.dump(selected_segments, file)

    preview_path = save_combined_preview(
        selected_segments,
        train_product_idx,
        product_eng_name,
        cam_name,
        out_visualized_dir,
    )

    blur_thresholds_by_class = {
        cls_id: metadata["main_threshold"]
        for cls_id, metadata in class_pool_metadata.items()
    }
    primary_blur_threshold = (
        next(iter(blur_thresholds_by_class.values()))
        if len(blur_thresholds_by_class) == 1
        else None
    )

    segment_data = {
        "cam": cam_name,
        "view": "top" if cam_name == "cam_0" else "side",
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "matched_count": len(matched),
        # 기존 의미를 유지: main blur pool 후보 수
        "candidates_count": filter_stats["main_candidates"],
        "total_eligible_candidates_count": len(purity_passed),
        "main_candidates_count": filter_stats["main_candidates"],
        "reserve_candidates_count": filter_stats["reserve_candidates"],
        "selected_count": len(selected_segments),
        "segments": selected_segments,
        "pkl_path": pkl_path,
        "preview_path": preview_path,
        # 단일 클래스 실행에서는 기존 float 필드와 호환됩니다.
        "blur_threshold_used": primary_blur_threshold,
        "blur_thresholds_by_class": blur_thresholds_by_class,
        "blur_percentile_used": BLUR_PERCENTILE,
        "reserve_blur_percentiles_used": _parse_reserve_percentiles(BLUR_PERCENTILE),
        "similarity_threshold_used": DIVERSITY_SIM_THRESHOLD,
        "similarity_spread_threshold_used": DIVERSITY_SIM_SPREAD_THRESHOLD,
        "saturation_patience_used": DIVERSITY_SATURATION_PATIENCE,
        "class_pool_metadata": class_pool_metadata,
        "class_selection_diagnostics": class_selection_diagnostics,
    }

    print(f"[segment_synthesis] {cam_name} segment 추출 완료")
    return segment_data

# =======================
# 📤 MinIO 업로드 함수
# =======================

def upload_to_minio(pkl_path: str, train_product_idx: str, product_eng_name: str,
                   endpoint: str, access_key: str, secret_key: str, 
                   bucket: str, base_prefix: str, secure: bool = False):
    """PKL 파일을 MinIO에 업로드"""
    if not MINIO_AVAILABLE:
        print("[warning] MinIO 모듈이 없어 업로드를 건너뜁니다.", file=sys.stderr)
        return False
    
    if not os.path.isfile(pkl_path):
        print(f"[error] PKL 파일이 없습니다: {pkl_path}", file=sys.stderr)
        return False
    
    try:
        # 날짜 형식: YYYYMMDD
        date_str = datetime.now().strftime("%Y%m%d")
        # 파일명: train_product_idx_engname_날짜.pkl
        pkl_filename = f"{train_product_idx}_{product_eng_name}_{date_str}.pkl"
        
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        
        prefix = base_prefix.strip("/")
        object_name = f"{prefix}/{pkl_filename}" if prefix else pkl_filename
        
        client.fput_object(bucket, object_name, pkl_path)
        print(f"[info] MinIO 업로드 완료: {bucket}/{object_name}")
        return True
    except Exception as e:
        print(f"[error] MinIO 업로드 실패: {e}", file=sys.stderr)
        return False

# =======================
# 🎯 메인 함수
# =======================

def main():
    parser = argparse.ArgumentParser(description="Segment 합성 스크립트")
    parser.add_argument("--img_dir", required=True, help="이미지 디렉토리")
    parser.add_argument("--lbl_dir", required=True, help="라벨 디렉토리")
    parser.add_argument("--out_segment_dir", required=True, help="Segment 출력 디렉토리")
    parser.add_argument("--out_mask_dir", required=True, help="Mask 출력 디렉토리")
    parser.add_argument("--out_visualized_dir", required=True, help="시각화 출력 디렉토리")
    parser.add_argument("--train_product_idx", required=True, help="train_product_idx")
    parser.add_argument("--product_eng_name", required=True, help="product_eng_name")
    parser.add_argument("--division_idx", required=True, help="division_idx")
    parser.add_argument("--is_cold", default="FALSE", help="IS_COLD (TRUE/FALSE)")
    parser.add_argument("--use_side", action="store_true", help="cam_2 (side)도 사용")
    
    # MinIO 설정 (환경변수 또는 인자)
    parser.add_argument("--minio_endpoint", default=None, help="MinIO endpoint")
    parser.add_argument("--minio_access_key", default=None, help="MinIO access key")
    parser.add_argument("--minio_secret_key", default=None, help="MinIO secret key")
    parser.add_argument("--minio_bucket", default="chaiimage", help="MinIO bucket")
    parser.add_argument("--minio_base_prefix", default="productSeg", help="MinIO base prefix")
    parser.add_argument("--minio_secure", default="false", help="MinIO secure (true/false)")
    parser.add_argument("--skip_minio_upload", action="store_true", help="상위 스크립트가 업로드할 때 내부 업로드를 건너뜀")
    
    # SAM 설정
    parser.add_argument("--sam_checkpoint", default=None, help="SAM checkpoint path")
    parser.add_argument("--sam_model_type", default=None, help="SAM model type")
    
    args = parser.parse_args()
    
    # SAM 설정 적용
    global SAM_CHECKPOINT, SAM_MODEL_TYPE
    if args.sam_checkpoint:
        SAM_CHECKPOINT = args.sam_checkpoint
    if args.sam_model_type:
        SAM_MODEL_TYPE = args.sam_model_type
    
    # 모든 segment 산출물은 호출자가 넘긴 output tree 아래에 저장합니다.
    visualized_dir = args.out_visualized_dir
    
    # 출력 디렉토리 생성
    os.makedirs(args.out_segment_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir, exist_ok=True)
    os.makedirs(visualized_dir, exist_ok=True)
    
    print("==========================================")
    print("Segment 합성 시작")
    print("==========================================")
    print(f"이미지 디렉토리: {args.img_dir}")
    print(f"라벨 디렉토리: {args.lbl_dir}")
    print(f"train_product_idx: {args.train_product_idx}")
    print(f"product_eng_name: {args.product_eng_name}")
    print(f"division_idx: {args.division_idx}")
    print(f"is_cold: {args.is_cold}")
    print(f"use_side: {args.use_side}")
    print(f"SAM checkpoint: {SAM_CHECKPOINT}")
    print(f"SAM model type: {SAM_MODEL_TYPE}")
    print(f"Device: {DEVICE}")
    print("==========================================")
    
    # SAM 모델 로드
    print("[segment_synthesis] SAM 모델 로딩 중...")
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"[error] SAM checkpoint를 찾을 수 없습니다: {SAM_CHECKPOINT}", file=sys.stderr)
        return 1
    
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    print("[segment_synthesis] SAM 모델 로딩 완료")
    
    # cam_0 (top) segment 추출
    cam_0_data = extract_segments_for_cam(
        args.img_dir, args.lbl_dir, "cam_0", predictor,
        args.out_segment_dir, args.out_mask_dir, visualized_dir,
        args.train_product_idx, args.product_eng_name
    )

    # side segment 추출 (선택적)
    # 리그마다 side 카메라 번호가 다르다 (예: _C division은 cam_2, _F division은 cam_1).
    # 이미지 디렉터리에서 cam_0 이외의 카메라를 찾아 side 로 사용한다.
    side_cam = None
    side_data = None
    if args.use_side:
        side_cam = detect_side_cam(args.img_dir)
        if side_cam:
            print(f"[segment_synthesis] side 카메라 자동 탐지: {side_cam}")
            side_data = extract_segments_for_cam(
                args.img_dir, args.lbl_dir, side_cam, predictor,
                args.out_segment_dir, args.out_mask_dir, visualized_dir,
                args.train_product_idx, args.product_eng_name
            )
        else:
            print("[segment_synthesis] side 카메라를 찾지 못했습니다 — cam_0 만 사용합니다")

    # 전체 결과 통합
    #
    # 카메라 데이터는 실제 카메라 이름(cam_0, cam_1, cam_2 …)을 키로 저장한다.
    # 시점(top/side)은 키에 담지 않는다 — 소비처가 cam_0 을 top,
    # 그 외 cam_N 을 side 로 해석한다. 리그마다 side 번호가 달라도
    # 스키마를 바꿀 필요가 없다.
    final_data = {
        "train_product_idx": args.train_product_idx,
        "product_eng_name": args.product_eng_name,
        "division_idx": args.division_idx,
        "is_cold": args.is_cold == "TRUE",
        "timestamp": datetime.now().isoformat(),
        "cam_0": cam_0_data,
    }
    if side_cam and side_data is not None:
        final_data[side_cam] = side_data
    
    # 최종 PKL 파일 저장
    date_str = datetime.now().strftime("%Y%m%d")
    pkl_filename = f"{args.train_product_idx}_{args.product_eng_name}_{date_str}.pkl"
    pkl_path = Path(args.out_segment_dir) / pkl_filename
    
    with open(pkl_path, "wb") as f:
        pickle.dump(final_data, f)
    
    print(f"[segment_synthesis] 최종 PKL 파일 저장 완료: {pkl_path}")
    
    # MinIO 업로드
    minio_endpoint = args.minio_endpoint or os.getenv("MINIO_ENDPOINT")
    minio_access_key = args.minio_access_key or os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = args.minio_secret_key or os.getenv("MINIO_SECRET_KEY")
    minio_bucket = args.minio_bucket or os.getenv("MINIO_BUCKET", "chaiimage")
    minio_base_prefix = args.minio_base_prefix or os.getenv("MINIO_BASE_PREFIX", "productSeg")
    minio_secure = str(args.minio_secure).lower() == "true" or os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    if args.skip_minio_upload:
        print("[info] segment_synthesis 내부 MinIO 업로드는 건너뜁니다. 상위 스크립트에서 1회 업로드합니다.")
    elif minio_endpoint and minio_access_key and minio_secret_key:
        print("")
        print("==========================================")
        print("MinIO 업로드 시작")
        print("==========================================")
        success = upload_to_minio(
            str(pkl_path), args.train_product_idx, args.product_eng_name,
            minio_endpoint, minio_access_key, minio_secret_key,
            minio_bucket, minio_base_prefix, minio_secure
        )
        if success:
            print("[info] MinIO 업로드 완료")
        else:
            print("[warn] MinIO 업로드 실패")
    else:
        print("[info] segment_synthesis 내부 MinIO 업로드는 건너뜁니다. 상위 스크립트에서 별도 업로드할 수 있습니다.")
    
    print("")
    print("==========================================")
    print("Segment 합성 완료")
    print("==========================================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())