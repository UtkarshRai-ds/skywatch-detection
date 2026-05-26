"""Top-level script: validate the NWPU VHR-10 dataset and save quality report.

Runs the structural loader and all five quality checks, then writes the
combined results to data/dataset_stats.json.

Usage::

    python run_loader.py
    python run_loader.py --data-dir path/to/custom/raw

Output::

    data/dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.data.loader import load_dataset_stats
from src.data.quality import run_all_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "raw"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "dataset_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the NWPU VHR-10 YOLO dataset and save stats to JSON."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the raw dataset directory (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the JSON report (default: data/dataset_stats.json).",
    )
    return parser.parse_args()


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    output_path: Path = args.output.resolve()

    # ── 1. Structural validation and file counts ──────────────────────────────
    _print_section("STEP 1 — Dataset Loader")
    logger.info("Loading dataset from: %s", data_dir)

    try:
        stats = load_dataset_stats(data_dir)
    except FileNotFoundError as exc:
        logger.error("Dataset validation failed: %s", exc)
        return 1

    print(f"\n  Total images : {stats['total_images']}")
    print(f"  Total labels : {stats['total_labels']}")
    for split, counts in stats["file_counts"].items():
        print(f"  {split:8s}: {counts['images']} images, {counts['labels']} labels")

    print(f"\n  Classes in data.yaml : {stats['yaml_config']['nc']}")
    if stats["yaml_config"]["names"]:
        for i, name in enumerate(stats["yaml_config"]["names"]):
            print(f"    [{i}] {name}")

    print(f"\n  Class IDs found in labels : {stats['class_ids_found']}")
    print(f"  All 10 classes present    : {stats['all_classes_present']}")
    if stats["missing_class_ids"]:
        print(f"  Missing class IDs         : {stats['missing_class_ids']}")

    # ── 2. Quality checks ─────────────────────────────────────────────────────
    _print_section("STEP 2 — Quality Checks")
    logger.info("Running quality checks…")

    quality_report = run_all_checks(data_dir, stats)

    for result in quality_report["results"]:
        icon = "PASS" if result["passed"] else "FAIL"
        print(f"\n  [{icon}] {result['check']}")
        print(f"        {result['summary']}")

    print(
        f"\n  Overall: {quality_report['checks_passed']}/{quality_report['checks_total']} checks passed"
    )

    # ── 3. Combine and serialise ──────────────────────────────────────────────
    _print_section("STEP 3 — Saving Report")

    combined: dict = {
        "loader": stats,
        "quality": quality_report,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Report saved to: %s", output_path)
    print(f"\n  Saved: {output_path}\n")

    return 0 if quality_report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
