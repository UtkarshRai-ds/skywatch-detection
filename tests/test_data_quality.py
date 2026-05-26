"""Tests for src/data/quality.py — the five dataset quality checks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.quality import (
    check_annotation_parity,
    check_bbox_validity,
    check_class_coverage,
    check_image_counts,
    check_image_readability,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_dataset(tmp_path: Path) -> Path:
    """One 64×64 JPEG + one valid label per split (class 0 only)."""
    for split in ("train", "valid", "test"):
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img001.jpg"), frame)
        (lbl_dir / "img001.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def valid_file_counts() -> dict:
    """file_counts dict that meets MIN_COUNTS for all splits."""
    return {
        "train": {"images": 560, "labels": 560},
        "valid": {"images": 160, "labels": 160},
        "test":  {"images": 80,  "labels": 80},
    }


# ── Each check returns a dict with a 'passed' key ─────────────────────────────

def test_check_image_counts_has_passed_key(tmp_path: Path, valid_file_counts: dict) -> None:
    result = check_image_counts(tmp_path, valid_file_counts)
    assert isinstance(result, dict)
    assert "passed" in result


def test_check_annotation_parity_has_passed_key(minimal_dataset: Path) -> None:
    # Minimal dataset: 1 image + 1 label per split — counts match
    file_counts = {s: {"images": 1, "labels": 1} for s in ("train", "valid", "test")}
    result = check_annotation_parity(minimal_dataset, file_counts)
    assert isinstance(result, dict)
    assert "passed" in result


def test_check_class_coverage_has_passed_key(tmp_path: Path) -> None:
    result = check_class_coverage(tmp_path, list(range(10)))
    assert isinstance(result, dict)
    assert "passed" in result


def test_check_image_readability_has_passed_key(minimal_dataset: Path) -> None:
    result = check_image_readability(minimal_dataset)
    assert isinstance(result, dict)
    assert "passed" in result


def test_check_bbox_validity_has_passed_key(minimal_dataset: Path) -> None:
    result = check_bbox_validity(minimal_dataset)
    assert isinstance(result, dict)
    assert "passed" in result


# ── check_image_counts ─────────────────────────────────────────────────────────

def test_image_count_check_passes_with_sufficient_counts(
    tmp_path: Path, valid_file_counts: dict
) -> None:
    result = check_image_counts(tmp_path, valid_file_counts)
    assert result["passed"] is True
    for split in ("train", "valid", "test"):
        assert result["details"][split]["passed"] is True


def test_image_count_check_fails_when_train_below_minimum(tmp_path: Path) -> None:
    low_counts = {
        "train": {"images": 10, "labels": 10},
        "valid": {"images": 160, "labels": 160},
        "test":  {"images": 80,  "labels": 80},
    }
    result = check_image_counts(tmp_path, low_counts)
    assert result["passed"] is False
    assert result["details"]["train"]["passed"] is False


# ── check_annotation_parity ────────────────────────────────────────────────────

def test_annotation_parity_passes_on_matched_dataset(minimal_dataset: Path) -> None:
    file_counts = {s: {"images": 1, "labels": 1} for s in ("train", "valid", "test")}
    result = check_annotation_parity(minimal_dataset, file_counts)
    assert result["passed"] is True


def test_annotation_parity_fails_on_orphan_image(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)

    # Add an image with no matching label
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "train" / "images" / "orphan.jpg"), frame)

    file_counts = {s: {"images": 0, "labels": 0} for s in ("train", "valid", "test")}
    file_counts["train"]["images"] = 1
    result = check_annotation_parity(tmp_path, file_counts)
    assert result["passed"] is False
    assert "orphan" in result["details"]["train"]["images_without_labels"]


# ── check_class_coverage ───────────────────────────────────────────────────────

def test_class_coverage_passes_with_all_10_classes(tmp_path: Path) -> None:
    result = check_class_coverage(tmp_path, list(range(10)))
    assert result["passed"] is True
    assert result["details"]["missing_class_ids"] == []


def test_class_coverage_detects_missing_classes(tmp_path: Path) -> None:
    # Only classes 0–4 present; classes 5–9 are missing
    result = check_class_coverage(tmp_path, [0, 1, 2, 3, 4])
    assert result["passed"] is False
    assert len(result["details"]["missing_class_ids"]) == 5
    assert 5 in result["details"]["missing_class_ids"]
    assert 9 in result["details"]["missing_class_ids"]


# ── check_bbox_validity ────────────────────────────────────────────────────────

def test_bbox_validity_passes_on_valid_labels(minimal_dataset: Path) -> None:
    result = check_bbox_validity(minimal_dataset)
    assert result["passed"] is True
    assert result["details"]["invalid_count"] == 0


def test_bbox_validity_catches_out_of_range_coordinate(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        (tmp_path / split / "labels").mkdir(parents=True)

    # cx = 1.5 is outside (0, 1]
    (tmp_path / "train" / "labels" / "bad.txt").write_text(
        "0 1.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    result = check_bbox_validity(tmp_path)
    assert result["passed"] is False
    assert result["details"]["invalid_count"] >= 1


def test_bbox_validity_catches_bbox_extending_beyond_image(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        (tmp_path / split / "labels").mkdir(parents=True)

    # cx=0.9, w=0.5 → cx + w/2 = 1.15 > 1.0
    (tmp_path / "train" / "labels" / "overflow.txt").write_text(
        "0 0.9 0.5 0.5 0.3\n", encoding="utf-8"
    )
    result = check_bbox_validity(tmp_path)
    assert result["passed"] is False


def test_bbox_validity_catches_non_numeric_value(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        (tmp_path / split / "labels").mkdir(parents=True)

    (tmp_path / "train" / "labels" / "text.txt").write_text(
        "0 abc 0.5 0.3 0.3\n", encoding="utf-8"
    )
    result = check_bbox_validity(tmp_path)
    assert result["passed"] is False
