"""Dataset loader and structural validator for NWPU VHR-10 in YOLO format.

Validates folder layout, counts files per split, verifies class coverage, and
returns a unified stats dict suitable for JSON serialisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Canonical NWPU VHR-10 class names in index order (0-9).
# The Roboflow-exported data.yaml contains citation text in the `names` field
# rather than the actual class labels, so we override it here.
NWPU_CLASSES: list[str] = [
    "airplane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "vehicle",
]

SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_NC: int = 10
IMAGE_EXTENSIONS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")


def validate_structure(data_dir: Path) -> dict[str, bool]:
    """Check that all required subdirectories and files exist.

    Args:
        data_dir: Root directory of the raw dataset.

    Returns:
        Mapping of path label → bool indicating existence.
    """
    checks: dict[str, bool] = {}
    for split in SPLITS:
        checks[f"{split}/images"] = (data_dir / split / "images").is_dir()
        checks[f"{split}/labels"] = (data_dir / split / "labels").is_dir()
    checks["data.yaml"] = (data_dir / "data.yaml").is_file()
    return checks


def count_files(data_dir: Path) -> dict[str, dict[str, int]]:
    """Count image and label files in each split directory.

    Args:
        data_dir: Root directory of the raw dataset.

    Returns:
        Nested dict: {split: {"images": int, "labels": int}}.
    """
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"

        images: list[Path] = []
        for pattern in IMAGE_EXTENSIONS:
            images.extend(img_dir.glob(pattern))

        labels = list(lbl_dir.glob("*.txt"))
        counts[split] = {"images": len(images), "labels": len(labels)}

    return counts


def load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """Load data.yaml and apply the canonical NWPU class names.

    The Roboflow export embeds citation paragraphs in the ``names`` field.
    We detect this by checking ``nc == 10`` and replace the names list with
    the authoritative NWPU_CLASSES constant.

    Args:
        yaml_path: Path to data.yaml.

    Returns:
        Parsed YAML dict with corrected ``names``.
    """
    with yaml_path.open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if cfg.get("nc") == EXPECTED_NC:
        cfg["names"] = NWPU_CLASSES

    return cfg


def collect_class_ids(data_dir: Path) -> set[int]:
    """Scan all label files and collect every class ID that appears.

    Args:
        data_dir: Root directory of the raw dataset.

    Returns:
        Set of integer class IDs found across all splits.
    """
    ids: set[int] = set()
    for split in SPLITS:
        lbl_dir = data_dir / split / "labels"
        for label_file in lbl_dir.glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    ids.add(int(parts[0]))
    return ids


def load_dataset_stats(data_dir: str | Path) -> dict[str, Any]:
    """Validate the dataset and return a comprehensive statistics dictionary.

    Performs structural validation, file counting, YAML loading, and class
    coverage checks. Raises ``FileNotFoundError`` if the expected directory
    layout is missing.

    Args:
        data_dir: Path to the raw dataset root (contains train/, valid/,
            test/, and data.yaml).

    Returns:
        Dict with the following keys:

        - ``data_dir``: Resolved absolute path (str).
        - ``structure_valid``: True when all expected paths exist.
        - ``structure_checks``: Per-path existence results.
        - ``file_counts``: Image and label counts per split.
        - ``total_images``: Sum across all splits.
        - ``total_labels``: Sum across all splits.
        - ``yaml_config``: Cleaned subset of data.yaml.
        - ``class_ids_found``: Sorted list of class IDs present in labels.
        - ``all_classes_present``: True when all 10 classes appear.
        - ``missing_class_ids``: Any expected class IDs absent from labels.

    Raises:
        FileNotFoundError: If any required subdirectory or data.yaml is absent.
    """
    data_dir = Path(data_dir)

    structure = validate_structure(data_dir)
    missing = [path for path, ok in structure.items() if not ok]
    if missing:
        raise FileNotFoundError(
            f"Dataset at '{data_dir}' is missing required paths: {missing}"
        )

    file_counts = count_files(data_dir)
    yaml_cfg = load_yaml_config(data_dir / "data.yaml")
    class_ids = collect_class_ids(data_dir)

    expected_ids = set(range(EXPECTED_NC))
    missing_ids = expected_ids - class_ids

    return {
        "data_dir": str(data_dir.resolve()),
        "structure_valid": True,
        "structure_checks": structure,
        "file_counts": file_counts,
        "total_images": sum(v["images"] for v in file_counts.values()),
        "total_labels": sum(v["labels"] for v in file_counts.values()),
        "yaml_config": {
            "nc": yaml_cfg.get("nc"),
            "names": yaml_cfg.get("names"),
            "train_path": yaml_cfg.get("train"),
            "val_path": yaml_cfg.get("val"),
            "test_path": yaml_cfg.get("test"),
        },
        "class_ids_found": sorted(class_ids),
        "all_classes_present": len(missing_ids) == 0,
        "missing_class_ids": sorted(missing_ids),
    }
