"""Top-level script: run all EDA functions and update data/dataset_stats.json.

Reads the existing dataset_stats.json produced by run_loader.py and appends
an ``eda`` key containing the output of all seven analysis functions. The
notebook (notebooks/eda.ipynb) reads from this JSON rather than re-scanning
the labels, keeping visualisation cells fast and reproducible.

Usage::

    python run_eda.py
    python run_eda.py --data-dir path/to/raw --output path/to/stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.features.analyze import (
    bbox_analysis,
    class_distribution,
    class_imbalance,
    cooccurrence_matrix,
    image_resolution,
    negative_images,
    spatial_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "raw"
DEFAULT_STATS = Path(__file__).parent / "data" / "dataset_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA on NWPU VHR-10 and append results to dataset_stats.json."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS)
    return parser.parse_args()


def _section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print("-" * 60)


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    stats_path: Path = args.output.resolve()

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return 1

    existing: dict = {}
    if stats_path.exists():
        existing = json.loads(stats_path.read_text(encoding="utf-8"))
        logger.info("Loaded existing stats from %s", stats_path)

    eda_results: dict = {}

    # ── 1. Class distribution ────────────────────────────────────────────────
    _section("1 / 7 — Class Distribution")
    t0 = time.perf_counter()
    dist = class_distribution(data_dir)
    eda_results["class_distribution"] = dist
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print(f"\n  Total annotations : {dist['total_annotations']}")
    for name in dist["class_order"]:
        n = dist["instances_per_class"][name]
        img = dist["images_per_class"][name]
        avg = dist["avg_instances_per_image"][name]
        print(f"  {name:<22}  {n:5d} instances  {img:3d} images  {avg:.2f} avg/img")

    # ── 2. Class imbalance ───────────────────────────────────────────────────
    _section("2 / 7 — Class Imbalance")
    t0 = time.perf_counter()
    imb = class_imbalance(data_dir)
    eda_results["class_imbalance"] = imb
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print(f"\n  Imbalance ratio   : {imb['imbalance_ratio']}×")
    print(f"  Dominant class    : {imb['dominant_class']}")
    print(f"  Rarest class      : {imb['rarest_class']}")
    print(f"  Undertrained      : {imb['undertrained_classes']}")

    # ── 3. BBox analysis ────────────────────────────────────────────────────
    _section("3 / 7 — BBox Geometry")
    t0 = time.perf_counter()
    bbox = bbox_analysis(data_dir)
    eda_results["bbox_analysis"] = bbox
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    ws = bbox["width_stats"]
    hs = bbox["height_stats"]
    as_ = bbox["area_stats"]
    print(f"\n  Width  : mean={ws['mean']:.4f}  median={ws['median']:.4f}  max={ws['max']:.4f}")
    print(f"  Height : mean={hs['mean']:.4f}  median={hs['median']:.4f}  max={hs['max']:.4f}")
    print(f"  Area   : mean={as_['mean']:.5f}  median={as_['median']:.6f}  (fraction of image)")
    print("\n  Median area by class:")
    for name, area in sorted(bbox["per_class_median_area"].items(), key=lambda x: -x[1]):
        print(f"    {name:<22}  {area:.6f}")

    # ── 4. Spatial heatmap ───────────────────────────────────────────────────
    _section("4 / 7 — Spatial Heatmap")
    t0 = time.perf_counter()
    spatial = spatial_heatmap(data_dir)
    eda_results["spatial_heatmap"] = spatial
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print(f"\n  CX mean={spatial['cx_stats']['mean']:.4f}  std={spatial['cx_stats']['std']:.4f}")
    print(f"  CY mean={spatial['cy_stats']['mean']:.4f}  std={spatial['cy_stats']['std']:.4f}")
    print(f"  Top-4 classes: {spatial['top4_classes']}")
    for name, c in spatial["per_class_centers"].items():
        print(f"    {name:<22}  cx_mean={c['cx_mean']:.3f}  cy_mean={c['cy_mean']:.3f}")

    # ── 5. Co-occurrence matrix ──────────────────────────────────────────────
    _section("5 / 7 — Co-occurrence Matrix")
    t0 = time.perf_counter()
    cooc = cooccurrence_matrix(data_dir)
    eda_results["cooccurrence_matrix"] = cooc
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print("\n  Top 5 co-occurring pairs (off-diagonal):")
    for pair in cooc["top_pairs"][:5]:
        print(f"    {pair['class_a']:<22} + {pair['class_b']:<22}  {pair['count']:5d}")

    # ── 6. Image resolution ──────────────────────────────────────────────────
    _section("6 / 7 — Image Resolution")
    t0 = time.perf_counter()
    res = image_resolution(data_dir)
    eda_results["image_resolution"] = res
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print(f"\n  Sample size            : {res['sample_size']}")
    print(f"  All same resolution    : {res['all_same_resolution']}")
    print(f"  Dominant resolution    : {res['dominant_resolution']}")
    print(f"  Unique resolutions     : {list(res['unique_resolutions'].keys())}")

    # ── 7. Negative images ───────────────────────────────────────────────────
    _section("7 / 7 — Negative Images")
    t0 = time.perf_counter()
    neg = negative_images(data_dir)
    eda_results["negative_images"] = neg
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    print(f"\n  Negative image count   : {neg['count']}")
    print(f"  Total label files      : {neg['total_label_files']}")
    print(f"  Note: {neg['note']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    _section("Saving dataset_stats.json")
    existing["eda"] = eda_results
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    logger.info("Updated: %s", stats_path)
    print(f"\n  Saved: {stats_path}")
    print(f"  Top-level keys: {list(existing.keys())}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
