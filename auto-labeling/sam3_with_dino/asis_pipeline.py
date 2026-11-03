#!/usr/bin/env python3
"""Candidate AS-IS auto-annotation pipeline with a run_product-compatible CLI.

This is the production entry point used by ``annotation_job.py``. It wraps the
validated inference engine and performs the complete two-pass flow through one
entry point:

1. new hand prompts + four hand seeds (two time ranges x occluded/clear),
2. calculate SAM3-mask/DINO-box fill ratios from pass 1,
3. rerun four-seed hand tracking with the 0.45 fill-ratio gate,
4. remove large edge-touching hand boxes and short hand-only runs,
5. export the final labels to the existing web-review COCO JSON format.

The public arguments used by ``annotation_job.run_annotation_job`` are passed
through to the validated ``run_product.py``.  AS-IS-only options use the
``--asis-`` prefix so this script can later replace ``SAM3_DINO_RUNNER`` without
changing the API/job layer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
HAND_CLASS_ID = "0"

DEFAULT_HAND_PROMPTS = "human hand|left hand|right hand|palm|fingers"
DEFAULT_SAM3_HAND_PROMPT = "human hand and fingers"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"

DEFAULT_ENGINE_RUNNER = SCRIPT_DIR / "run_product.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated AS-IS two-pass annotation pipeline. Unknown "
            "run_product options are forwarded to the inference engine."
        )
    )
    # Arguments consumed by the current annotation_job/run_product contract.
    parser.add_argument("--img-root", default=str(PROJECT_ROOT / "img_download"))
    parser.add_argument("--runs-root", default=str(PROJECT_ROOT / "runs" / "sam3_with_dino"))
    parser.add_argument("--product", action="append", default=[])
    parser.add_argument("--cams", nargs="+", default=["cam_0", "cam_2"])
    parser.add_argument("--run-name", default=time.strftime("asis_%Y%m%d_%H%M%S"))
    parser.add_argument("--backend", choices=("video",), default="video")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--product-class-id", type=int)
    parser.add_argument("--product-name")
    parser.add_argument("--product-prompts", default="")
    parser.add_argument("--hand-prompts", default=DEFAULT_HAND_PROMPTS)
    parser.add_argument("--sam3-hand-prompt", default=DEFAULT_SAM3_HAND_PROMPT)
    parser.add_argument("--background-reference-root", default="")
    parser.add_argument("--root-folder", default="")
    parser.add_argument("--product-idx", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-export", action="store_true")

    # Candidate-only controls. Defaults reproduce the selected ff:0.45 mode.
    parser.add_argument("--asis-engine-runner", type=Path)
    parser.add_argument(
        "--asis-work-dir",
        type=Path,
        help="Isolate intermediate caches, logs and summary for parallel product jobs",
    )
    parser.add_argument(
        "--asis-fill-threshold",
        type=float,
        default=float(os.getenv("SAM3_DINO_ASIS_FILL_THRESHOLD", "0.45")),
    )
    parser.add_argument(
        "--asis-edge-area-ratio",
        type=float,
        default=float(os.getenv("SAM3_DINO_ASIS_EDGE_AREA_RATIO", "0.30")),
    )
    parser.add_argument(
        "--asis-short-run",
        type=int,
        default=int(os.getenv("SAM3_DINO_ASIS_SHORT_RUN", "3")),
    )
    parser.add_argument(
        "--asis-dino-batch",
        type=int,
        default=int(os.getenv("SAM3_DINO_ASIS_DINO_BATCH", "4")),
    )
    return parser


def _resolve_engine(explicit: Path | None) -> Path:
    candidate = explicit or DEFAULT_ENGINE_RUNNER
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"validated AS-IS run_product.py not found: {candidate}")


def _resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def _product_dirs(args: argparse.Namespace) -> list[Path]:
    img_root = _resolve_path(args.img_root)
    if args.product:
        return [
            Path(item) if Path(item).is_absolute() else img_root / item
            for item in args.product
        ]
    if not img_root.is_dir():
        return []
    return sorted(path for path in img_root.iterdir() if path.is_dir() and not path.name.startswith("_"))


def _strip_wrapper_options(argv: list[str]) -> list[str]:
    """Remove options owned by this wrapper before forwarding argv."""
    takes_value = {
        "--run-name",
        "--runs-root",
        "--asis-engine-runner",
        "--asis-work-dir",
        "--asis-fill-threshold",
        "--asis-edge-area-ratio",
        "--asis-short-run",
        "--asis-dino-batch",
        "--hand-prompts",
        "--sam3-hand-prompt",
    }
    flags = {"--skip-export"}
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]
        if name in flags:
            index += 1
            continue
        if name in takes_value:
            index += 1 if "=" in token else 2
            continue
        output.append(token)
        index += 1
    return output


def _run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {log_path}")


def _engine_command(
    args: argparse.Namespace,
    passthrough: list[str],
    engine: Path,
    runs_root: Path,
    run_name: str,
    extra: list[str],
) -> list[str]:
    return [
        str(Path(args.python_bin)),
        str(engine),
        *passthrough,
        "--runs-root",
        str(runs_root),
        "--run-name",
        run_name,
        "--skip-export",
        "--sam3-hand-prompt",
        args.sam3_hand_prompt,
        "--hand-prompts",
        args.hand_prompts,
        "--dino-model",
        args.dino_model,
        *extra,
    ]


def _image_dir(product_dir: Path, cam: str) -> Path:
    root = product_dir / cam
    nested = root / "images"
    return nested if nested.is_dir() else root


def _source_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _polygon_mask_area(path: Path, width: int, height: int) -> int:
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    found = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < 7 or fields[0] != HAND_CLASS_ID:
            continue
        values = fields[1:]
        points = [
            (float(values[i]) * width, float(values[i + 1]) * height)
            for i in range(0, len(values) - 1, 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
            found = True
    return int(np.asarray(mask, dtype=np.uint8).sum()) if found else 0


def _write_fill_file(
    debug_jsonl: Path,
    seg_dir: Path,
    image_dir: Path,
    output: Path,
) -> int:
    ratios: dict[str, float] = {}
    for line in debug_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stem = Path(str(record.get("image", ""))).stem
        hand = record.get("best_hand")
        if not stem or not hand:
            continue
        box = hand.get("box_xyxy") or []
        if len(box) != 4:
            continue
        box_area = max(0.0, float(box[2]) - float(box[0])) * max(
            0.0, float(box[3]) - float(box[1])
        )
        if box_area <= 1.0:
            continue
        image_path = _source_image(image_dir, stem)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        seg_path = seg_dir / f"{stem}.txt"
        mask_area = _polygon_mask_area(seg_path, width, height) if seg_path.is_file() else 0
        ratios[stem] = float(mask_area) / box_area
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ratios, ensure_ascii=True), encoding="utf-8")
    return len(ratios)


def _strip_hand(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not (line.split() and line.split()[0] == HAND_CLASS_ID)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _filter_large_edge_hands(labels_dir: Path, image_dir: Path, area_ratio: float) -> int:
    removed = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _source_image(image_dir, label_path.stem)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        kept: list[str] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0] != HAND_CLASS_ID:
                kept.append(line)
                continue
            center_x, center_y, box_w, box_h = (float(value) for value in fields[1:5])
            x1 = (center_x - box_w / 2.0) * width
            y1 = (center_y - box_h / 2.0) * height
            x2 = (center_x + box_w / 2.0) * width
            y2 = (center_y + box_h / 2.0) * height
            touches_edge = x1 < 3.0 or y1 < 3.0 or x2 > width - 3.0 or y2 > height - 3.0
            if touches_edge and box_w * box_h > area_ratio:
                removed += 1
                continue
            kept.append(line)
        label_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def _has_hand(path: Path) -> bool:
    return any(
        fields and fields[0] == HAND_CLASS_ID
        for fields in (
            line.split()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        )
    )


def _filter_short_hand_runs(labels_dir: Path, maximum: int) -> int:
    def frame_key(path: Path) -> tuple[int, str]:
        numbers = re.findall(r"\d+", path.stem)
        return (int(numbers[-1]) if numbers else -1, path.stem)

    files = sorted(labels_dir.glob("*.txt"), key=frame_key)
    runs: list[list[Path]] = []
    current: list[Path] = []
    for path in files:
        if _has_hand(path):
            current.append(path)
        else:
            if current and len(current) <= maximum:
                runs.append(current)
            current = []
    if current and len(current) <= maximum:
        runs.append(current)
    removed = 0
    for run in runs:
        for path in run:
            path.write_text(_strip_hand(path.read_text(encoding="utf-8")), encoding="utf-8")
            removed += 1
    return removed


def _export_command(
    args: argparse.Namespace,
    exporter: Path,
    product_dir: Path,
    product_root: Path,
    cams: list[str],
) -> list[str]:
    command = [
        str(Path(args.python_bin)),
        str(exporter),
        "--results-root",
        str(product_root),
        "--image-root",
        str(product_dir),
        "--cams",
        *cams,
    ]
    if args.product_class_id is not None:
        command.extend(["--product-class-id", str(args.product_class_id)])
    if args.product_name:
        command.extend(["--product-name", args.product_name])
    if args.root_folder:
        command.extend(["--root-folder", args.root_folder])
    if args.product_idx:
        command.extend(["--product-idx", args.product_idx])
    return command


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, _ = _parser().parse_known_args(raw_argv)
    if not 0.0 <= args.asis_fill_threshold <= 1.0:
        raise SystemExit("--asis-fill-threshold must be between 0 and 1")
    if not 0.0 <= args.asis_edge_area_ratio <= 1.0:
        raise SystemExit("--asis-edge-area-ratio must be between 0 and 1")
    if args.asis_short_run < 0:
        raise SystemExit("--asis-short-run must be >= 0")
    if args.asis_dino_batch < 1:
        raise SystemExit("--asis-dino-batch must be >= 1")

    engine = _resolve_engine(args.asis_engine_runner)
    exporter = engine.parent / "export_web_annotations.py"
    if not exporter.is_file():
        raise FileNotFoundError(f"AS-IS exporter not found: {exporter}")

    runs_root = _resolve_path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    pass1_name = f"{args.run_name}__pass1"
    work_dir = _resolve_path(args.asis_work_dir) if args.asis_work_dir else None
    cache_root = work_dir / "cache" if work_dir else runs_root / "_asis_cache" / args.run_name
    pass1_log = work_dir / "pass1.log" if work_dir else runs_root / pass1_name / "asis_pipeline.log"
    pass2_log = work_dir / "pass2.log" if work_dir else runs_root / args.run_name / "asis_pipeline.log"
    dino_cache_root = cache_root / "dino"
    fill_root = cache_root / "fill"
    product_cache_root = cache_root / "product"
    for directory in (dino_cache_root, fill_root, product_cache_root):
        directory.mkdir(parents=True, exist_ok=True)

    products = _product_dirs(args)
    if not products:
        raise SystemExit("no product directories selected")

    passthrough = _strip_wrapper_options(raw_argv)
    env = os.environ.copy()
    env.setdefault("TORCH_COMPILE_DISABLE", "1")

    common_asis = [
        "--hand-seed-dual",
        "--hand-seed-halves",
        "2",
        "--hand-recovery-max-anchors",
        "0",
        "--dino-batch",
        str(args.asis_dino_batch),
        "--product-cache-root",
        str(product_cache_root),
    ]
    pass1 = _engine_command(
        args,
        passthrough,
        engine,
        runs_root,
        pass1_name,
        common_asis,
    )
    print(f"[asis] pass1 start run={pass1_name}", flush=True)
    _run(pass1, pass1_log, env)

    fill_counts: dict[str, int] = {}
    for product_dir in products:
        product_id = product_dir.name.split("_", 1)[0]
        available_cams = [cam for cam in args.cams if (product_dir / cam).is_dir()]
        for cam in available_cams:
            pass1_cam = runs_root / pass1_name / product_dir.name / cam
            debug_jsonl = pass1_cam / "debug" / "detections.jsonl"
            seg_dir = pass1_cam / "labels_seg"
            if not debug_jsonl.is_file() or not seg_dir.is_dir():
                raise FileNotFoundError(f"incomplete AS-IS pass1 output: {pass1_cam}")
            cache_name = f"{product_id}_{cam}.jsonl"
            shutil.copy2(debug_jsonl, dino_cache_root / cache_name)
            fill_path = fill_root / f"{product_id}_{cam}.json"
            count = _write_fill_file(debug_jsonl, seg_dir, _image_dir(product_dir, cam), fill_path)
            fill_counts[f"{product_dir.name}/{cam}"] = count
            print(f"[asis] fill {product_dir.name}/{cam}: {count} frames", flush=True)

    pass2 = _engine_command(
        args,
        passthrough,
        engine,
        runs_root,
        args.run_name,
        [
            *common_asis,
            "--hand-seed-fill-min",
            str(args.asis_fill_threshold),
            "--hand-seed-fill-root",
            str(fill_root),
            "--dino-cache-root",
            str(dino_cache_root),
        ],
    )
    print(f"[asis] pass2 start run={args.run_name}", flush=True)
    _run(pass2, pass2_log, env)

    postprocess: dict[str, dict[str, int]] = {}
    for product_dir in products:
        product_root = runs_root / args.run_name / product_dir.name
        available_cams = [cam for cam in args.cams if (product_dir / cam).is_dir()]
        for cam in available_cams:
            labels_dir = product_root / cam / "labels"
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"missing AS-IS pass2 labels: {labels_dir}")
            edge_removed = _filter_large_edge_hands(
                labels_dir,
                _image_dir(product_dir, cam),
                args.asis_edge_area_ratio,
            )
            short_removed = _filter_short_hand_runs(labels_dir, args.asis_short_run)
            postprocess[f"{product_dir.name}/{cam}"] = {
                "large_edge_hands_removed": edge_removed,
                "short_run_frames_removed": short_removed,
            }
            print(
                f"[asis] postprocess {product_dir.name}/{cam}: "
                f"edge={edge_removed} short={short_removed}",
                flush=True,
            )
        if not args.skip_export:
            _run(
                _export_command(args, exporter, product_dir, product_root, available_cams),
                product_root / "out" / "export_web_annotations.log",
                env,
            )

    summary = {
        "pipeline": "new_prompt+four_seeds+fill_two_pass+background_filter+temporal_consistency",
        "engine": str(engine),
        "run_name": args.run_name,
        "pass1_run_name": pass1_name,
        "fill_threshold": args.asis_fill_threshold,
        "edge_area_ratio": args.asis_edge_area_ratio,
        "short_run": args.asis_short_run,
        "hand_prompts": args.hand_prompts,
        "sam3_hand_prompt": args.sam3_hand_prompt,
        "fill_frames": fill_counts,
        "postprocess": postprocess,
    }
    summary_path = work_dir / "asis_summary.json" if work_dir else runs_root / args.run_name / "asis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[asis] complete summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
