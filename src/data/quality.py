"""Data quality checks for the NWPU VHR-10 YOLO dataset.

Runs five independent checks and aggregates them into a single report dict:

1. Image counts per split — verifies counts meet expected minimums.
2. Annotation parity — every image has a matching label file (by stem).
3. Class coverage — all 10 NWPU classes appear at least once.
4. Image readability — a random sample of images can be decoded by OpenCV.
5. Bbox coordinate validity — all YOLO coordinates are in the [0, 1] range.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2

from src.data.loader import (
    EXPECTED_NC,
    NWPU_CLASSES,
    SPLITS,
    IMAGE_EXTENSIONS,
)

# Minimum expected image counts per split based on the NWPU VHR-10 distribution.
MIN_COUNTS: dict[str, int] = {"train": 400, "valid": 100, "test": 50}

# Fraction of images to sample for the readability and bbox checks.
READABILITY_SAMPLE_FRACTION: float = 0.1
READABILITY_MIN_SAMPLE: int = 10


def _collect_image_paths(split_dir: Path) -> list[Path]:
    """Return all image paths under a split's images/ subdirectory."""
    paths: list[Path] = []
    img_dir = split_dir / "images"
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(img_dir.glob(pattern))
    return paths


def check_image_counts(
    data_dir: Path, file_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """Check 1: Verify image counts per split meet expected minimums.

    Args:
        data_dir: Dataset root (unused here, kept for uniform signature).
        file_counts: Output of ``loader.count_files``.

    Returns:
        Check result dict with keys ``passed``, ``details``, and ``summary``.
    """
    details: dict[str, Any] = {}
    all_passed = True

    for split in SPLITS:
        count = file_counts[split]["images"]
        minimum = MIN_COUNTS[split]
        ok = count >= minimum
        details[split] = {
            "count": count,
            "minimum_expected": minimum,
            "passed": ok,
        }
        if not ok:
            all_passed = False

    return {
        "check": "image_counts",
        "passed": all_passed,
        "details": details,
        "summary": (
            "All splits meet minimum image count thresholds."
            if all_passed
            else "One or more splits have fewer images than expected."
        ),
    }


def check_annotation_parity(
    data_dir: Path, file_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """Check 2: Verify every image file has a corresponding label file by stem.

    Matches image stems (filename without extension) to label stems. Reports
    images without annotations and labels without corresponding images.

    Args:
        data_dir: Dataset root directory.
        file_counts: Output of ``loader.count_files`` (used for fast summary).

    Returns:
        Check result dict with per-split orphan lists.
    """
    details: dict[str, Any] = {}
    all_passed = True

    for split in SPLITS:
        split_dir = data_dir / split
        img_stems = {p.stem for p in _collect_image_paths(split_dir)}
        lbl_stems = {p.stem for p in (split_dir / "labels").glob("*.txt")}

        images_without_labels = sorted(img_stems - lbl_stems)
        labels_without_images = sorted(lbl_stems - img_stems)
        ok = not images_without_labels and not labels_without_images

        details[split] = {
            "image_count": len(img_stems),
            "label_count": len(lbl_stems),
            "images_without_labels": images_without_labels,
            "labels_without_images": labels_without_images,
            "passed": ok,
        }
        if not ok:
            all_passed = False

    return {
        "check": "annotation_parity",
        "passed": all_passed,
        "details": details,
        "summary": (
            "Every image has a matching label file."
            if all_passed
            else "Mismatched image/label files found (see details)."
        ),
    }


def check_class_coverage(
    data_dir: Path, class_ids_found: list[int]
) -> dict[str, Any]:
    """Check 3: Verify all 10 NWPU classes appear at least once in labels.

    Args:
        data_dir: Dataset root (unused, kept for uniform signature).
        class_ids_found: Sorted list of class IDs from ``loader.collect_class_ids``.

    Returns:
        Check result dict listing present and missing classes.
    """
    found_set = set(class_ids_found)
    expected_set = set(range(EXPECTED_NC))
    missing_ids = sorted(expected_set - found_set)

    present_classes = [NWPU_CLASSES[i] for i in sorted(found_set) if i < EXPECTED_NC]
    missing_classes = [NWPU_CLASSES[i] for i in missing_ids if i < EXPECTED_NC]

    passed = len(missing_ids) == 0

    return {
        "check": "class_coverage",
        "passed": passed,
        "details": {
            "expected_classes": EXPECTED_NC,
            "found_class_count": len(found_set),
            "present_classes": present_classes,
            "missing_classes": missing_classes,
            "missing_class_ids": missing_ids,
        },
        "summary": (
            f"All {EXPECTED_NC} classes present."
            if passed
            else f"Missing {len(missing_ids)} class(es): {missing_classes}"
        ),
    }


def check_image_readability(data_dir: Path) -> dict[str, Any]:
    """Check 4: Sample images across splits and verify OpenCV can decode them.

    Samples ``READABILITY_SAMPLE_FRACTION`` of images per split (at least
    ``READABILITY_MIN_SAMPLE`` when the split is large enough). Reports any
    file that ``cv2.imread`` cannot open (returns None).

    Args:
        data_dir: Dataset root directory.

    Returns:
        Check result dict with counts of readable/unreadable images and a list
        of any corrupt file paths.
    """
    corrupt: list[str] = []
    sampled_total = 0

    for split in SPLITS:
        all_images = _collect_image_paths(data_dir / split)
        k = max(READABILITY_MIN_SAMPLE, int(len(all_images) * READABILITY_SAMPLE_FRACTION))
        k = min(k, len(all_images))
        sample = random.sample(all_images, k)
        sampled_total += k

        for img_path in sample:
            frame = cv2.imread(str(img_path))
            if frame is None:
                corrupt.append(str(img_path))

    passed = len(corrupt) == 0

    return {
        "check": "image_readability",
        "passed": passed,
        "details": {
            "images_sampled": sampled_total,
            "corrupt_images": corrupt,
            "corrupt_count": len(corrupt),
        },
        "summary": (
            f"All {sampled_total} sampled images are readable."
            if passed
            else f"{len(corrupt)} image(s) could not be decoded by OpenCV."
        ),
    }


def check_bbox_validity(data_dir: Path) -> dict[str, Any]:
    """Check 5: Validate YOLO bbox coordinates are within the [0, 1] range.

    Each label line must have 5 fields: class_id cx cy w h. All four
    coordinate values must be in (0, 1]. Additionally checks that the box
    does not extend beyond the image boundary (cx ± w/2 and cy ± h/2
    must both be in [0, 1]).

    Args:
        data_dir: Dataset root directory.

    Returns:
        Check result dict with counts of valid/invalid annotations and
        examples of any invalid lines.
    """
    invalid_examples: list[dict[str, Any]] = []
    total_annotations = 0
    invalid_count = 0

    for split in SPLITS:
        lbl_dir = data_dir / split / "labels"
        for label_file in sorted(lbl_dir.glob("*.txt")):
            for lineno, raw in enumerate(
                label_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw.strip()
                if not line:
                    continue
                total_annotations += 1
                parts = line.split()

                if len(parts) != 5:
                    invalid_count += 1
                    if len(invalid_examples) < 20:
                        invalid_examples.append(
                            {
                                "file": str(label_file),
                                "line": lineno,
                                "content": line,
                                "reason": f"expected 5 fields, got {len(parts)}",
                            }
                        )
                    continue

                try:
                    _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    invalid_count += 1
                    if len(invalid_examples) < 20:
                        invalid_examples.append(
                            {
                                "file": str(label_file),
                                "line": lineno,
                                "content": line,
                                "reason": "non-numeric coordinate value",
                            }
                        )
                    continue

                coords_in_range = all(0.0 < v <= 1.0 for v in (cx, cy, w, h))
                bbox_in_image = (
                    0.0 <= cx - w / 2 and cx + w / 2 <= 1.0
                    and 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0
                )

                if not (coords_in_range and bbox_in_image):
                    invalid_count += 1
                    if len(invalid_examples) < 20:
                        reason_parts = []
                        if not coords_in_range:
                            reason_parts.append("coordinate outside (0, 1]")
                        if not bbox_in_image:
                            reason_parts.append("bbox extends beyond image boundary")
                        invalid_examples.append(
                            {
                                "file": str(label_file),
                                "line": lineno,
                                "content": line,
                                "reason": "; ".join(reason_parts),
                            }
                        )

    passed = invalid_count == 0

    return {
        "check": "bbox_validity",
        "passed": passed,
        "details": {
            "total_annotations": total_annotations,
            "invalid_count": invalid_count,
            "invalid_examples": invalid_examples,
        },
        "summary": (
            f"All {total_annotations} annotations have valid YOLO coordinates."
            if passed
            else f"{invalid_count}/{total_annotations} annotations have invalid coordinates."
        ),
    }


def run_all_checks(
    data_dir: str | Path,
    loader_stats: dict[str, Any],
) -> dict[str, Any]:
    """Run all five quality checks and return a consolidated report.

    Args:
        data_dir: Dataset root directory (str or Path).
        loader_stats: Output of ``loader.load_dataset_stats``.

    Returns:
        Dict with keys ``all_passed`` (bool), ``checks_passed`` (int),
        ``checks_total`` (int), and ``results`` (list of per-check dicts).
    """
    data_dir = Path(data_dir)
    file_counts: dict[str, dict[str, int]] = loader_stats["file_counts"]
    class_ids_found: list[int] = loader_stats["class_ids_found"]

    results = [
        check_image_counts(data_dir, file_counts),
        check_annotation_parity(data_dir, file_counts),
        check_class_coverage(data_dir, class_ids_found),
        check_image_readability(data_dir),
        check_bbox_validity(data_dir),
    ]

    passed_count = sum(1 for r in results if r["passed"])

    return {
        "all_passed": passed_count == len(results),
        "checks_passed": passed_count,
        "checks_total": len(results),
        "results": results,
    }
