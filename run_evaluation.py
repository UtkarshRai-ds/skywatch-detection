"""Top-level evaluation script: build class_stats.json and training_history.json.

Loads both YOLOv8 checkpoints, structures the hardcoded test-set metrics,
compares the two models, runs inference on 15 test images, and writes two
output files:

  - ``data/class_stats.json``     — per-class AP, comparison, prediction summary
  - ``data/training_history.json`` — epoch-by-epoch metrics parsed from CSVs

Usage::

    python run_evaluation.py
    python run_evaluation.py --models-dir models --data-dir data --output-dir data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.models.evaluate import (
    build_test_results,
    get_model_info,
    measure_inference_speed,
    parse_training_csv,
)
from src.models.compare import (
    compare_models,
    identify_challenging_classes,
    speed_accuracy_tradeoff,
)
from src.models.predict import run_predictions, summarise_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Hardcoded test-set results from the Colab training run ────────────────────
# Source: model.val(split='test') executed on Colab T4 after 50-epoch training.

_TEST_RESULTS: dict[str, dict] = {
    "yolov8n": {
        "overall": {
            "mAP50":     0.689,
            "precision": 0.794,
            "recall":    0.634,
            "mAP50_95":  0.426,
        },
        "per_class": {
            "airplane":            0.561,
            "ship":                0.782,
            "storage tank":        0.895,
            "baseball diamond":    0.892,
            "tennis court":        0.838,
            "basketball court":    0.430,
            "ground track field":  0.579,
            "harbor":              0.928,
            "bridge":              0.030,
            "vehicle":             0.952,
        },
    },
    "yolov8s": {
        "overall": {
            "mAP50":     0.697,
            "precision": 0.743,
            "recall":    0.690,
            "mAP50_95":  0.437,
        },
        "per_class": {
            "airplane":            0.675,
            "ship":                0.764,
            "storage tank":        0.926,
            "baseball diamond":    0.901,
            "tennis court":        0.841,
            "basketball court":    0.422,
            "ground track field":  0.589,
            "harbor":              0.889,
            "bridge":              0.027,
            "vehicle":             0.934,
        },
    },
}

DEFAULT_MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_DATA_DIR   = Path(__file__).parent / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 models and save JSON reports.")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--data-dir",   type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--skip-predict", action="store_true",
                   help="Skip sample detection generation (faster).")
    return p.parse_args()


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def build_training_history(data_dir: Path) -> dict:
    """Parse both results CSVs and return a combined training history dict."""
    history: dict = {}
    for name in ("yolov8n", "yolov8s"):
        csv_path = data_dir / f"{name}_results.csv"
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            continue

        rows = parse_training_csv(csv_path)
        best_row = max(rows, key=lambda r: r.get("metrics/mAP50(B)", 0.0))
        total_time_s = rows[-1].get("time", 0.0) if rows else 0.0

        history[name] = {
            "epochs":        len(rows),
            "best_epoch":    int(best_row.get("epoch", 0)),
            "best_val_mAP50": round(best_row.get("metrics/mAP50(B)", 0.0), 4),
            "total_time_s":  round(total_time_s, 1),
            "total_time_min": round(total_time_s / 60, 1),
            "history": [
                {
                    "epoch":          int(r.get("epoch", 0)),
                    "time_s":         round(r.get("time", 0.0), 2),
                    "train_box_loss": round(r.get("train/box_loss", 0.0), 5),
                    "train_cls_loss": round(r.get("train/cls_loss", 0.0), 5),
                    "train_dfl_loss": round(r.get("train/dfl_loss", 0.0), 5),
                    "val_box_loss":   round(r.get("val/box_loss",   0.0), 5),
                    "val_cls_loss":   round(r.get("val/cls_loss",   0.0), 5),
                    "val_dfl_loss":   round(r.get("val/dfl_loss",   0.0), 5),
                    "precision":      round(r.get("metrics/precision(B)", 0.0), 5),
                    "recall":         round(r.get("metrics/recall(B)",    0.0), 5),
                    "mAP50":          round(r.get("metrics/mAP50(B)",     0.0), 5),
                    "mAP50_95":       round(r.get("metrics/mAP50-95(B)",  0.0), 5),
                    "lr":             round(r.get("lr/pg0", 0.0), 8),
                }
                for r in rows
            ],
        }

    return history


def main() -> int:
    args = parse_args()
    models_dir: Path = args.models_dir.resolve()
    data_dir:   Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    # ── 1. Load model metadata ────────────────────────────────────────────────
    _banner("1 / 5 — Model Metadata")
    model_infos: dict[str, Any] = {}
    for name, fname in (("yolov8n", "yolov8n_best.pt"), ("yolov8s", "yolov8s_best.pt")):
        pt = models_dir / fname
        info = get_model_info(pt)
        model_infos[name] = info
        print(f"\n  {info.architecture}")
        print(f"    Parameters : {info.params:,}")
        print(f"    Task       : {info.task}")

    # ── 2. Speed benchmark ────────────────────────────────────────────────────
    _banner("2 / 5 — Inference Speed (CPU)")
    test_images = sorted((data_dir / "raw" / "test" / "images").glob("*.jpg"))[:10]
    speed_results: dict[str, dict] = {}

    for name, info in model_infos.items():
        t0 = time.perf_counter()
        spd = measure_inference_speed(Path(info.path), test_images, n_warmup=2)
        elapsed = time.perf_counter() - t0
        speed_results[name] = spd
        print(f"\n  {name}")
        print(f"    Inference (avg) : {spd['inference_ms']:.1f} ms")
        print(f"    Total (avg)     : {spd['total_ms']:.1f} ms  ({spd['fps']:.1f} FPS)")
        print(f"    Benchmark time  : {elapsed:.1f}s")

    # ── 3. Build structured test results ─────────────────────────────────────
    _banner("3 / 5 — Test-Set Evaluation Results")
    csv_best_epochs: dict[str, int] = {}
    for name in ("yolov8n", "yolov8s"):
        csv_path = data_dir / f"{name}_results.csv"
        if csv_path.exists():
            rows = parse_training_csv(csv_path)
            best = max(rows, key=lambda r: r.get("metrics/mAP50(B)", 0.0))
            csv_best_epochs[name] = int(best.get("epoch", 0))

    all_results: dict[str, dict] = {}
    for name in ("yolov8n", "yolov8s"):
        tr = _TEST_RESULTS[name]
        res = build_test_results(
            model_info=model_infos[name],
            overall=tr["overall"],
            per_class=tr["per_class"],
            speed=speed_results.get(name),
            best_epoch=csv_best_epochs.get(name),
        )
        all_results[name] = res
        ov = res["overall"]
        print(f"\n  {name.upper()}")
        print(f"    mAP50={ov['mAP50']:.3f}  P={ov['precision']:.3f}  R={ov['recall']:.3f}  mAP50-95={ov['mAP50_95']:.3f}")
        print(f"    Best class : {res['best_class']}  ({res['per_class_mAP50'][res['best_class']]:.3f})")
        print(f"    Worst class: {res['worst_class']} ({res['per_class_mAP50'][res['worst_class']]:.3f})")

    # ── 4. Comparison ─────────────────────────────────────────────────────────
    _banner("4 / 5 — Model Comparison")
    comparison = compare_models(all_results["yolov8n"], all_results["yolov8s"])
    tradeoff   = speed_accuracy_tradeoff(list(all_results.values()))
    challenges = identify_challenging_classes(list(all_results.values()), threshold=0.5)

    print(f"\n  Winner (mAP50)  : {comparison['winner_mAP50']}")
    print(f"  mAP50 delta     : {comparison['overall_delta']['mAP50']['delta']:+.4f}")
    print(f"  Param ratio     : {comparison['param_ratio']}x")
    print(f"  Challenging     : {challenges['challenging_classes']}")
    print(f"\n  {comparison['recommendation']}")

    print("\n  Per-class deltas (s − n):")
    for cls in comparison["sorted_by_impact"]:
        d = comparison["per_class_delta"][cls]
        print(f"    {cls:<25}  {d['delta']:+.4f}  (n={d['yolov8n']:.3f}  s={d['yolov8s']:.3f})")

    # ── 5. Sample detections ──────────────────────────────────────────────────
    pred_summary: dict[str, dict] = {}
    if not args.skip_predict:
        _banner("5 / 5 — Sample Detections (15 images per model)")
        sample_dir = data_dir / "sample_detections"
        test_img_dir = data_dir / "raw" / "test" / "images"

        for name, pt_fname in (("yolov8n", "yolov8n_best.pt"), ("yolov8s", "yolov8s_best.pt")):
            pt = models_dir / pt_fname
            logs = run_predictions(
                model_path=pt,
                image_dir=test_img_dir,
                output_dir=sample_dir,
                n_images=15,
                conf=0.25,
            )
            summary = summarise_predictions(logs)
            pred_summary[name] = summary
            print(f"\n  {name}  →  {summary['total_detections']} detections "
                  f"across {summary['images_processed']} images  "
                  f"(avg {summary['avg_detections_per_image']:.1f}/image)")
    else:
        _banner("5 / 5 — Sample Detections (skipped)")

    # ── Save class_stats.json ─────────────────────────────────────────────────
    _banner("Saving JSON Reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_stats: dict = {
        "models": all_results,
        "comparison": comparison,
        "speed_accuracy_tradeoff": tradeoff,
        "challenging_classes": challenges,
        "prediction_summary": pred_summary,
    }
    cs_path = output_dir / "class_stats.json"
    cs_path.write_text(json.dumps(class_stats, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", cs_path)

    # ── Save training_history.json ────────────────────────────────────────────
    history = build_training_history(data_dir)
    th_path = output_dir / "training_history.json"
    th_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Saved: %s", th_path)

    print(f"\n  class_stats.json     : {cs_path}")
    print(f"  training_history.json: {th_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
