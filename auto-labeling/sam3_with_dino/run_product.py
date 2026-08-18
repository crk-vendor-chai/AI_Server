#!/usr/bin/env python3
"""Run Grounding DINO + SAM3 annotation for product camera folders."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(sys.executable)
BACKGROUND_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def class_id_from_folder(folder_name: str) -> int:
    try:
        return int(folder_name.split("_", 1)[0])
    except Exception:
        return 1


def strip_foldername_timestamp(foldername: str) -> str:
    name = re.sub(r"_\d{8}_\d{6}$", "", str(foldername or "").strip())
    return re.sub(r"^\d+_", "", name)


def product_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    img_root = Path(args.img_root)
    if not img_root.is_absolute():
        img_root = PROJECT_ROOT / img_root
    if args.product:
        return [Path(p) if Path(p).is_absolute() else img_root / p for p in args.product]
    return sorted(p for p in img_root.iterdir() if p.is_dir())


def build_annotation_command(args: argparse.Namespace, product_dir: Path, cam: str, task_root: Path) -> list[str]:
    product_class_id = int(args.product_class_id or class_id_from_folder(product_dir.name))
    product_name = args.product_name or strip_foldername_timestamp(product_dir.name)
    annotator = "video_annotate.py" if args.backend == "video" else "auto_annotate.py"
    cmd = [
        str(Path(args.python_bin)),
        str(SCRIPT_DIR / annotator),
        "--img_dir", str(product_dir / cam),
        "--out_label_dir", str(task_root / "labels"),
        "--out_seg_label_dir", str(task_root / "labels_seg"),
        "--out_visualized_dir", str(task_root / "visualized"),
        "--out_debug_jsonl", str(task_root / "debug" / "detections.jsonl"),
        "--summary_path", str(task_root / "summary.txt"),
        "--product_class_id", str(product_class_id),
        "--product_name", product_name,
        "--dino_model", args.dino_model,
        "--dino_device", "cuda" if args.gpu else args.device,
        "--sam3_device", "cuda" if args.gpu else args.device,
        "--sam3_hand_prompt", args.sam3_hand_prompt,
        "--hand_box_threshold", str(args.hand_box_threshold),
        "--object_box_threshold", str(args.object_box_threshold),
        "--text_threshold", str(args.text_threshold),
        "--sam3_confidence", str(args.sam3_confidence),
        "--sam3_area_weight", str(args.sam3_area_weight),
        "--sam3_min_prompt_iou", str(args.sam3_min_prompt_iou),
        "--sam3_min_area_ratio", str(args.sam3_min_area_ratio),
        "--sam3_max_area_ratio", str(args.sam3_max_area_ratio),
        "--temporal_object_iou_weight", str(args.temporal_object_iou_weight),
        "--temporal_decay", str(args.temporal_decay),
        "--temporal_max_gap", str(args.temporal_max_gap),
        "--min_temporal_compatibility", str(args.min_temporal_compatibility),
        "--min_hand_object_relation", str(args.min_hand_object_relation),
        "--object_solo_threshold", str(args.object_solo_threshold),
        "--hand_solo_threshold", str(args.hand_solo_threshold),
        "--segmentation_fallback", args.segmentation_fallback,
        "--max_hand_candidates", str(args.max_hand_candidates),
        "--max_object_candidates", str(args.max_object_candidates),
        "--nms_iou", str(args.nms_iou),
        "--min_hand_area", str(args.min_hand_area),
        "--max_hand_area", str(args.max_hand_area),
        "--min_object_area", str(args.min_object_area),
        "--max_object_area", str(args.max_object_area),
        "--min_hand_aspect", str(args.min_hand_aspect),
        "--max_hand_aspect", str(args.max_hand_aspect),
        "--min_object_aspect", str(args.min_object_aspect),
        "--max_object_aspect", str(args.max_object_aspect),
        "--max_hand_width", str(args.max_hand_width),
        "--max_hand_height", str(args.max_hand_height),
        "--max_object_width", str(args.max_object_width),
        "--max_object_height", str(args.max_object_height),
        "--background_pixel_threshold", str(args.background_pixel_threshold),
        "--background_min_hand_foreground", str(args.background_min_hand_foreground),
        "--background_min_object_foreground", str(args.background_min_object_foreground),
        "--compact_hand_max_product_gap", str(args.compact_hand_max_product_gap),
        "--compact_hand_cluster_radius", str(args.compact_hand_cluster_radius),
        "--compact_hand_score_window", str(args.compact_hand_score_window),
        "--compact_hand_min_margin", str(args.compact_hand_min_margin),
        "--compact_hand_expansion", str(args.compact_hand_expansion),
        "--hand_mask_support_expansion", str(args.hand_mask_support_expansion),
    ]
    if args.backend == "video":
        cmd.extend([
            "--video_product_seed_threshold", str(args.video_product_seed_threshold),
            "--video_hand_seed_threshold", str(args.video_hand_seed_threshold),
            "--video_min_output_score", str(args.video_min_output_score),
            "--video_seed_support_radius", str(args.video_seed_support_radius),
            "--cross_class_duplicate_iou", str(args.cross_class_duplicate_iou),
            "--cross_class_duplicate_area_similarity", str(args.cross_class_duplicate_area_similarity),
            "--cross_class_duplicate_mask_iou", str(args.cross_class_duplicate_mask_iou),
            "--cross_class_hand_margin", str(args.cross_class_hand_margin),
            "--hand_partial_mask_support_expansion", str(args.hand_partial_mask_support_expansion),
            "--hand_max_mask_support_expansion", str(args.hand_max_mask_support_expansion),
            "--hand_component_min_area_ratio", str(args.hand_component_min_area_ratio),
            "--hand_component_max_gap", str(args.hand_component_max_gap),
            "--hand_component_product_gap", str(args.hand_component_product_gap),
            "--hand_collapse_area_ratio", str(args.hand_collapse_area_ratio),
            "--hand_collapse_dimension_ratio", str(args.hand_collapse_dimension_ratio),
            "--hand_collapse_history_min", str(args.hand_collapse_history_min),
            "--hand_collapse_history_window", str(args.hand_collapse_history_window),
            "--hand_recovery_min_area_ratio", str(args.hand_recovery_min_area_ratio),
            "--hand_recovery_search_radius", str(args.hand_recovery_search_radius),
            "--hand_recovery_context_frames", str(args.hand_recovery_context_frames),
            "--hand_recovery_max_anchors", str(args.hand_recovery_max_anchors),
            "--hand_seed_every", str(args.hand_seed_every),
            "--hand_seed_halves", str(args.hand_seed_halves),
            "--hand_seed_margin", str(args.hand_seed_margin),
            "--hand_seed_endcut", str(args.hand_seed_endcut),
            "--hand_seed_fill_min", str(args.hand_seed_fill_min),
            "--hand_seed_area_factor", str(args.hand_seed_area_factor),
            "--hand_point_arm_offset", str(args.hand_point_arm_offset),
            "--hand_recovery_merge_gap", str(args.hand_recovery_merge_gap),
            "--visible_hand_support_expansion", str(args.visible_hand_support_expansion),
            "--visible_hand_min_dino_overlap", str(args.visible_hand_min_dino_overlap),
            "--shoe_competitor_mode", args.shoe_competitor_mode,
            "--arm_competitor_mode", args.arm_competitor_mode,
            "--shoe_competitor_prompts", args.shoe_competitor_prompts,
            "--arm_competitor_prompts", args.arm_competitor_prompts,
            "--competitor_box_threshold", str(args.competitor_box_threshold),
            "--competitor_text_threshold", str(args.competitor_text_threshold),
            "--max_competitor_candidates", str(args.max_competitor_candidates),
            "--competitor_min_score", str(args.competitor_min_score),
            "--competitor_score_margin", str(args.competitor_score_margin),
            "--competitor_min_overlap", str(args.competitor_min_overlap),
            "--competitor_temporal_radius", str(args.competitor_temporal_radius),
            "--competitor_min_temporal_support", str(args.competitor_min_temporal_support),
            "--competitor_temporal_compatibility", str(args.competitor_temporal_compatibility),
        ])
        if args.dino_cache_root:
            dc = (Path(args.dino_cache_root) / f"{product_dir.name.split('_')[0]}_{cam}.jsonl")
            if dc.is_file():
                cmd.extend(["--dino_cache", str(dc)])
        if args.dino_batch > 1:
            cmd.extend(["--dino_batch", str(args.dino_batch)])
        if args.dino_autocast_dtype != "none":
            cmd.extend(["--dino_autocast_dtype", args.dino_autocast_dtype])
        if args.product_cache_root:
            # 있으면 불러 쓰고, 없으면 이번 실행이 만들어 둔다 (dino 캐시와 달리 조건 없이 넘긴다)
            pc = (Path(args.product_cache_root)
                  / f"{product_dir.name.split('_')[0]}_{cam}.npz")
            cmd.extend(["--product_cache", str(pc)])
        if args.hand_seed_fill_root:
            # <root>/<상품번호>_<카메라>.json 규칙으로 찾아 넘긴다
            fp = Path(args.hand_seed_fill_root) / f"{product_dir.name.split('_')[0]}_{cam}.json"
            if fp.is_file():
                cmd.extend(["--hand_seed_fill_file", str(fp)])
        if args.hand_point_refine:
            cmd.append("--hand_point_refine")
        if args.hand_seed_dual:
            cmd.append("--hand_seed_dual")
        if args.visible_hand_mode:
            cmd.append("--visible_hand_mode")
        if args.finger_seed_only:
            cmd.append("--finger_seed_only")
    if args.background_reference_root:
        reference_root = Path(args.background_reference_root)
        if not reference_root.is_absolute():
            reference_root = PROJECT_ROOT / reference_root
        camera_reference_dir = reference_root / cam
        if camera_reference_dir.is_dir():
            for reference in sorted(camera_reference_dir.iterdir()):
                if reference.is_file() and reference.suffix.lower() in BACKGROUND_EXTS:
                    cmd.extend(["--background_reference", str(reference)])
    if args.product_prompts:
        cmd.extend(["--product_prompts", args.product_prompts])
    if args.hand_prompts:
        cmd.extend(["--hand_prompts", args.hand_prompts])
    if args.start_index:
        cmd.extend(["--start_index", str(args.start_index)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def export_web_json(
    args: argparse.Namespace,
    product_dir: Path,
    product_root: Path,
    cams: list[str],
) -> int:
    product_class_id = int(args.product_class_id or class_id_from_folder(product_dir.name))
    product_name = args.product_name or strip_foldername_timestamp(product_dir.name)
    cmd = [
        str(Path(args.python_bin)),
        str(SCRIPT_DIR / "export_web_annotations.py"),
        "--results-root", str(product_root),
        "--image-root", str(product_dir),
        "--cams", *cams,
        "--product-class-id", str(product_class_id),
        "--product-name", product_name,
        "--train-product-idx", str(product_class_id),
    ]
    if args.root_folder:
        cmd.extend(["--root-folder", args.root_folder])
    if args.product_idx:
        cmd.extend(["--product-idx", args.product_idx])
    return run_command(cmd, product_root / "out" / "export_web_annotations.log", os.environ.copy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-root", default=str(PROJECT_ROOT / "img_download"))
    parser.add_argument("--runs-root", default=str(SCRIPT_DIR / "runs"))
    parser.add_argument("--product", action="append", default=[])
    parser.add_argument("--cams", nargs="+", default=["cam_0", "cam_2"])
    parser.add_argument("--run-name", default=time.strftime("dino_sam3_%Y%m%d_%H%M%S"))
    parser.add_argument("--backend", choices=["video", "image"], default="video")
    parser.add_argument("--hand-seed-every", type=int, default=0,
                        dest="hand_seed_every")
    parser.add_argument("--hand-seed-dual", action="store_true", dest="hand_seed_dual")
    parser.add_argument("--hand-seed-halves", type=int, default=2,
                        dest="hand_seed_halves")
    parser.add_argument("--hand-seed-margin", type=float, default=0.0,
                        dest="hand_seed_margin")
    parser.add_argument("--hand-seed-endcut", type=float, default=0.0,
                        dest="hand_seed_endcut")
    parser.add_argument("--hand-seed-fill-min", type=float, default=0.0,
                        dest="hand_seed_fill_min")
    parser.add_argument("--hand-seed-fill-root", default="",
                        dest="hand_seed_fill_root")
    parser.add_argument("--dino-cache-root", default="", dest="dino_cache_root")
    parser.add_argument("--product-cache-root", default="", dest="product_cache_root")
    parser.add_argument("--dino-batch", type=int, default=1, dest="dino_batch")
    parser.add_argument("--dino-autocast-dtype", choices=("bfloat16", "float16", "none"),
                        default="none", dest="dino_autocast_dtype")
    parser.add_argument("--hand-seed-area-factor", type=float, default=0.0,
                        dest="hand_seed_area_factor")
    parser.add_argument("--hand-point-refine", action="store_true", dest="hand_point_refine")
    parser.add_argument("--hand-point-arm-offset", type=float, default=0.35,
                        dest="hand_point_arm_offset")
    parser.add_argument("--gpu", default="1", help="Physical GPU id. Empty string uses --device without CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--product-class-id", type=int)
    parser.add_argument("--product-name")
    parser.add_argument("--product-prompts", default="")
    parser.add_argument("--hand-prompts", default="")
    parser.add_argument(
        "--sam3-hand-prompt",
        default="complete human hand from wrist to fingertips",
    )
    parser.add_argument("--background-reference-root", default="")
    parser.add_argument("--background-pixel-threshold", type=float, default=22.0)
    parser.add_argument("--background-min-hand-foreground", type=float, default=0.06)
    parser.add_argument("--background-min-object-foreground", type=float, default=0.08)
    parser.add_argument("--compact-hand-max-product-gap", type=float, default=0.18)
    parser.add_argument("--compact-hand-cluster-radius", type=float, default=0.22)
    parser.add_argument("--compact-hand-score-window", type=float, default=0.14)
    parser.add_argument("--compact-hand-min-margin", type=float, default=0.08)
    parser.add_argument("--compact-hand-expansion", type=float, default=0.30)
    parser.add_argument("--hand-mask-support-expansion", type=float, default=0.15)
    parser.add_argument("--hand-partial-mask-support-expansion", type=float, default=0.45)
    parser.add_argument("--hand-max-mask-support-expansion", type=float, default=0.60)
    parser.add_argument("--hand-component-min-area-ratio", type=float, default=0.04)
    parser.add_argument("--hand-component-max-gap", type=float, default=0.08)
    parser.add_argument("--hand-component-product-gap", type=float, default=0.05)
    parser.add_argument("--hand-collapse-area-ratio", type=float, default=0.45)
    parser.add_argument("--hand-collapse-dimension-ratio", type=float, default=0.50)
    parser.add_argument("--hand-collapse-history-min", type=int, default=4)
    parser.add_argument("--hand-collapse-history-window", type=int, default=9)
    parser.add_argument("--hand-recovery-min-area-ratio", type=float, default=0.65)
    parser.add_argument("--hand-recovery-search-radius", type=int, default=60)
    parser.add_argument("--hand-recovery-context-frames", type=int, default=45)
    parser.add_argument("--hand-recovery-max-anchors", type=int, default=0)
    parser.add_argument("--hand-recovery-merge-gap", type=int, default=5)
    parser.add_argument(
        "--visible-hand-mode",
        action="store_true",
        help=(
            "Experimental video policy that boxes only currently visible hand pixels "
            "and accepts finger-only observations."
        ),
    )
    parser.add_argument("--visible-hand-support-expansion", type=float, default=0.08)
    parser.add_argument("--visible-hand-min-dino-overlap", type=float, default=0.70)
    parser.add_argument("--finger-seed-only", action="store_true")
    parser.add_argument(
        "--shoe-competitor-mode", choices=("off", "shadow", "reject"), default="off"
    )
    parser.add_argument(
        "--arm-competitor-mode", choices=("off", "shadow", "reject"), default="off"
    )
    parser.add_argument(
        "--shoe-competitor-prompts",
        default="shoe|sneaker|footwear|lower leg and shoe",
    )
    parser.add_argument(
        "--arm-competitor-prompts",
        default="human forearm|human arm|sleeve|lower arm",
    )
    parser.add_argument("--competitor-box-threshold", type=float, default=0.25)
    parser.add_argument("--competitor-text-threshold", type=float, default=0.20)
    parser.add_argument("--max-competitor-candidates", type=int, default=10)
    parser.add_argument("--competitor-min-score", type=float, default=0.40)
    parser.add_argument("--competitor-score-margin", type=float, default=0.05)
    parser.add_argument("--competitor-min-overlap", type=float, default=0.50)
    parser.add_argument("--competitor-temporal-radius", type=int, default=2)
    parser.add_argument("--competitor-min-temporal-support", type=int, default=1)
    parser.add_argument("--competitor-temporal-compatibility", type=float, default=0.25)
    parser.add_argument("--hand-box-threshold", type=float, default=0.25)
    parser.add_argument("--object-box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--sam3-confidence", type=float, default=0.35)
    parser.add_argument("--sam3-area-weight", type=float, default=0.15)
    parser.add_argument("--sam3-min-prompt-iou", type=float, default=0.15)
    parser.add_argument("--sam3-min-area-ratio", type=float, default=0.15)
    parser.add_argument("--sam3-max-area-ratio", type=float, default=4.0)
    parser.add_argument("--temporal-object-iou-weight", type=float, default=0.20)
    parser.add_argument("--temporal-decay", type=float, default=0.75)
    parser.add_argument("--temporal-max-gap", type=int, default=5)
    parser.add_argument("--min-temporal-compatibility", type=float, default=0.30)
    parser.add_argument("--min-hand-object-relation", type=float, default=0.08)
    parser.add_argument("--object-solo-threshold", type=float, default=0.38)
    parser.add_argument("--hand-solo-threshold", type=float, default=0.40)
    parser.add_argument("--video-product-seed-threshold", type=float, default=0.24)
    parser.add_argument("--video-hand-seed-threshold", type=float, default=0.30)
    parser.add_argument("--video-min-output-score", type=float, default=0.40)
    parser.add_argument("--video-seed-support-radius", type=int, default=3)
    parser.add_argument("--cross-class-duplicate-iou", type=float, default=0.88)
    parser.add_argument("--cross-class-duplicate-area-similarity", type=float, default=0.85)
    parser.add_argument("--cross-class-duplicate-mask-iou", type=float, default=0.80)
    parser.add_argument("--cross-class-hand-margin", type=float, default=0.10)
    parser.add_argument("--segmentation-fallback", choices=["skip", "box"], default="skip")
    parser.add_argument("--max-hand-candidates", type=int, default=8)
    parser.add_argument("--max-object-candidates", type=int, default=10)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--min-hand-area", type=float, default=0.0)
    parser.add_argument("--max-hand-area", type=float, default=1.0)
    parser.add_argument("--min-object-area", type=float, default=0.0)
    parser.add_argument("--max-object-area", type=float, default=1.0)
    parser.add_argument("--min-hand-aspect", type=float, default=0.0)
    parser.add_argument("--max-hand-aspect", type=float, default=1e9)
    parser.add_argument("--min-object-aspect", type=float, default=0.0)
    parser.add_argument("--max-object-aspect", type=float, default=1e9)
    parser.add_argument("--max-hand-width", type=float, default=1.0)
    parser.add_argument("--max-hand-height", type=float, default=1.0)
    parser.add_argument("--max-object-width", type=float, default=1.0)
    parser.add_argument("--max-object-height", type=float, default=1.0)
    parser.add_argument("--root-folder", default="")
    parser.add_argument("--product-idx", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    run_root = runs_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SCRIPT_DIR)
        if not existing_pythonpath
        else str(SCRIPT_DIR) + os.pathsep + existing_pythonpath
    )
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    failed = 0
    for product_dir in product_dirs_from_args(args):
        if not product_dir.is_dir():
            print(f"[warn] missing product dir: {product_dir}")
            failed += 1
            continue
        product_root = run_root / product_dir.name
        available_cams = [cam for cam in args.cams if (product_dir / cam).is_dir()]
        if not available_cams:
            print(f"[error] no camera directories found: {product_dir}")
            failed += 1
            continue
        for cam in available_cams:
            cam_dir = product_dir / cam
            if not cam_dir.is_dir():
                print(f"[warn] missing cam dir: {cam_dir}")
                continue
            task_root = product_root / cam
            cmd = build_annotation_command(args, product_dir, cam, task_root)
            print(f"[run] {product_dir.name}/{cam} gpu={args.gpu or 'none'}")
            code = run_command(cmd, task_root / "run.log", env)
            if code != 0:
                print(f"[error] {product_dir.name}/{cam} failed: {task_root / 'run.log'}")
                failed += 1
                continue
            print(f"[done] {task_root / 'summary.txt'}")
        if not args.skip_export:
            code = export_web_json(args, product_dir, product_root, available_cams)
            if code != 0:
                print(f"[error] web json export failed: {product_root / 'out'}")
                failed += 1
            else:
                print(f"[done] {product_root / 'out' / 'annotation.json'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
