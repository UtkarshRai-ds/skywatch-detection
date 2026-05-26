"""Exploratory data analysis functions for the NWPU VHR-10 YOLO dataset.

All public functions accept a ``data_dir`` path and return plain dicts whose
values are Python scalars or nested lists — suitable for JSON serialisation and
direct use as Plotly trace inputs.

Typical usage::

    from pathlib import Path
    from src.features.analyze import (
        class_distribution,
        class_imbalance,
        bbox_analysis,
        spatial_heatmap,
        cooccurrence_matrix,
        image_resolution,
        negative_images,
    )

    data_dir = Path("data/raw")
    dist = class_distribution(data_dir)
    imb  = class_imbalance(data_dir)
    ...
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.loader import NWPU_CLASSES, SPLITS, IMAGE_EXTENSIONS

# ── Private helpers ────────────────────────────────────────────────────────────

_Annotation = dict[str, Any]  # keys: class_id, cx, cy, w, h, split, stem


def _parse_annotations(data_dir: Path) -> list[_Annotation]:
    """Parse every YOLO label file in all splits into a flat list of dicts.

    Each dict has keys: ``class_id`` (int), ``cx``, ``cy``, ``w``, ``h``
    (float), ``split`` (str), ``stem`` (str — image filename without extension).

    Args:
        data_dir: Root directory containing train/, valid/, test/ subdirs.

    Returns:
        List of annotation dicts for all objects across all splits.
    """
    annotations: list[_Annotation] = []
    for split in SPLITS:
        lbl_dir = data_dir / split / "labels"
        for lf in sorted(lbl_dir.glob("*.txt")):
            for raw in lf.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) == 5:
                    annotations.append(
                        {
                            "class_id": int(parts[0]),
                            "cx": float(parts[1]),
                            "cy": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                            "split": split,
                            "stem": lf.stem,
                        }
                    )
    return annotations


def _collect_label_files(data_dir: Path) -> list[tuple[str, Path]]:
    """Return (split, label_path) pairs for every label file across all splits."""
    pairs: list[tuple[str, Path]] = []
    for split in SPLITS:
        for lf in sorted((data_dir / split / "labels").glob("*.txt")):
            pairs.append((split, lf))
    return pairs


def _collect_image_paths(data_dir: Path) -> list[Path]:
    """Return all image paths across all splits."""
    paths: list[Path] = []
    for split in SPLITS:
        img_dir = data_dir / split / "images"
        for pattern in IMAGE_EXTENSIONS:
            paths.extend(img_dir.glob(pattern))
    return paths


def _stats_dict(values: list[float]) -> dict[str, float]:
    """Compute basic descriptive statistics for a numeric list."""
    arr = np.array(values, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


# ── Public analysis functions ──────────────────────────────────────────────────


def class_distribution(data_dir: str | Path) -> dict[str, Any]:
    """Compute per-class instance counts, image counts, and density statistics.

    Aggregates annotations from all splits (train + valid + test) to give a
    dataset-wide view of how instances are distributed across the 10 NWPU classes.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``instances_per_class``: {class_name: int} — total annotation count.
        - ``images_per_class``: {class_name: int} — unique images containing the class.
        - ``avg_instances_per_image``: {class_name: float} — density when class is present.
        - ``instances_per_image_stats``: descriptive stats for the per-image count.
        - ``total_annotations``: int.
        - ``class_order``: list[str] sorted by instance count descending.
    """
    data_dir = Path(data_dir)
    annotations = _parse_annotations(data_dir)

    instance_counts: Counter[int] = Counter()
    images_with_class: defaultdict[int, set[str]] = defaultdict(set)
    per_image_counts: Counter[str] = Counter()

    for ann in annotations:
        cid = ann["class_id"]
        instance_counts[cid] += 1
        images_with_class[cid].add(ann["stem"])
        per_image_counts[ann["stem"]] += 1

    instances_per_class: dict[str, int] = {}
    images_per_class: dict[str, int] = {}
    avg_instances_per_image: dict[str, float] = {}

    for i, name in enumerate(NWPU_CLASSES):
        n_inst = instance_counts.get(i, 0)
        n_img = len(images_with_class.get(i, set()))
        instances_per_class[name] = n_inst
        images_per_class[name] = n_img
        avg_instances_per_image[name] = round(n_inst / n_img, 3) if n_img > 0 else 0.0

    class_order = sorted(  # type: ignore[arg-type]
        instances_per_class, key=instances_per_class.get, reverse=True
    )

    return {
        "instances_per_class": instances_per_class,
        "images_per_class": images_per_class,
        "avg_instances_per_image": avg_instances_per_image,
        "instances_per_image_stats": _stats_dict(list(per_image_counts.values())),
        "total_annotations": len(annotations),
        "class_order": class_order,
    }


def class_imbalance(data_dir: str | Path) -> dict[str, Any]:
    """Measure class imbalance and flag classes at risk of under-training.

    Uses the effective number of samples (Cui et al., 2019) to quantify how
    much useful signal each class provides relative to the dominant class.
    Classes whose instance count falls below ``mean - 0.5 * std`` are flagged.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``imbalance_ratio``: float — max / min instance count.
        - ``dominant_class``: str.
        - ``rarest_class``: str.
        - ``effective_num_samples``: {class_name: float}.
        - ``undertrained_flags``: {class_name: bool}.
        - ``undertrained_classes``: list[str].
        - ``raw_counts``: {class_name: int} for quick bar plotting.
    """
    data_dir = Path(data_dir)
    dist = class_distribution(data_dir)
    counts: dict[str, int] = dist["instances_per_class"]

    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    rarest = min(counts, key=counts.get)  # type: ignore[arg-type]

    # Effective number of samples: N_eff = (1 - β^n) / (1 - β), β = (N-1)/N
    total = sum(counts.values())
    beta = (total - 1) / total
    effective: dict[str, float] = {
        name: round((1 - beta ** n) / (1 - beta), 2) if n > 0 else 0.0
        for name, n in counts.items()
    }

    arr = np.array(list(counts.values()), dtype=float)
    threshold = float(arr.mean() - 0.5 * arr.std())
    flags: dict[str, bool] = {name: counts[name] < threshold for name in counts}
    undertrained = [name for name, flagged in flags.items() if flagged]

    return {
        "imbalance_ratio": imbalance_ratio,
        "dominant_class": dominant,
        "rarest_class": rarest,
        "effective_num_samples": effective,
        "undertrained_flags": flags,
        "undertrained_classes": undertrained,
        "raw_counts": counts,
    }


def bbox_analysis(data_dir: str | Path) -> dict[str, Any]:
    """Analyse bounding box geometry across all annotations.

    All measurements are in normalised image units (0–1), so area represents
    the fraction of total image area the box occupies.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``width_stats``, ``height_stats``, ``area_stats``, ``aspect_ratio_stats``:
          descriptive stats dicts (min/max/mean/median/std/p25/p75).
        - ``per_class_median_area``: {class_name: float}.
        - ``per_class_median_ar``: {class_name: float} — median width/height ratio.
        - ``scatter_sample``: list of {w, h, area, ar, class_name} — up to 1000
          randomly sampled annotations for scatter plots.
    """
    data_dir = Path(data_dir)
    annotations = _parse_annotations(data_dir)

    widths, heights, areas, aspect_ratios = [], [], [], []
    per_class_w: defaultdict[int, list[float]] = defaultdict(list)
    per_class_h: defaultdict[int, list[float]] = defaultdict(list)

    for ann in annotations:
        w, h = ann["w"], ann["h"]
        ar = w / h if h > 0 else 0.0
        area = w * h
        widths.append(w)
        heights.append(h)
        areas.append(area)
        aspect_ratios.append(ar)
        per_class_w[ann["class_id"]].append(w)
        per_class_h[ann["class_id"]].append(h)

    per_class_median_area: dict[str, float] = {
        NWPU_CLASSES[i]: round(
            float(np.median(np.array(per_class_w[i]) * np.array(per_class_h[i]))), 6
        )
        for i in range(len(NWPU_CLASSES))
        if per_class_w[i]
    }
    per_class_median_ar: dict[str, float] = {
        NWPU_CLASSES[i]: round(
            float(
                np.median(np.array(per_class_w[i]) / np.maximum(np.array(per_class_h[i]), 1e-9))
            ), 4
        )
        for i in range(len(NWPU_CLASSES))
        if per_class_w[i]
    }

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(annotations), size=min(1000, len(annotations)), replace=False)
    scatter_sample = [
        {
            "w": round(annotations[i]["w"], 5),
            "h": round(annotations[i]["h"], 5),
            "area": round(annotations[i]["w"] * annotations[i]["h"], 6),
            "ar": round(
                annotations[i]["w"] / annotations[i]["h"] if annotations[i]["h"] > 0 else 0, 4
            ),
            "class_name": NWPU_CLASSES[annotations[i]["class_id"]],
        }
        for i in sample_idx.tolist()
    ]

    return {
        "width_stats": _stats_dict(widths),
        "height_stats": _stats_dict(heights),
        "area_stats": _stats_dict(areas),
        "aspect_ratio_stats": _stats_dict(aspect_ratios),
        "per_class_median_area": per_class_median_area,
        "per_class_median_ar": per_class_median_ar,
        "scatter_sample": scatter_sample,
    }


def spatial_heatmap(data_dir: str | Path) -> dict[str, Any]:
    """Compute bbox centre distributions for heatmap and per-class scatter plots.

    Returns cx/cy lists for all annotations and for the 4 most populous classes
    individually, enabling grid-level density and class-specific spatial plots.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``cx_all``, ``cy_all``: lists of all centre coordinates (floats).
        - ``cx_stats``, ``cy_stats``: descriptive statistics.
        - ``per_class_centers``: {class_name: {cx_mean, cy_mean, cx_list, cy_list}}
          for the 4 classes with the most instances.
        - ``top4_classes``: list of the 4 class names by instance count.
    """
    data_dir = Path(data_dir)
    annotations = _parse_annotations(data_dir)

    cx_all = [ann["cx"] for ann in annotations]
    cy_all = [ann["cy"] for ann in annotations]

    per_class_cx: defaultdict[int, list[float]] = defaultdict(list)
    per_class_cy: defaultdict[int, list[float]] = defaultdict(list)
    class_instance_counts: Counter[int] = Counter()

    for ann in annotations:
        cid = ann["class_id"]
        per_class_cx[cid].append(ann["cx"])
        per_class_cy[cid].append(ann["cy"])
        class_instance_counts[cid] += 1

    top4_ids = [cid for cid, _ in class_instance_counts.most_common(4)]
    top4_classes = [NWPU_CLASSES[i] for i in top4_ids]

    per_class_centers: dict[str, dict[str, Any]] = {}
    for cid in top4_ids:
        name = NWPU_CLASSES[cid]
        cxs = per_class_cx[cid]
        cys = per_class_cy[cid]
        per_class_centers[name] = {
            "cx_mean": round(float(np.mean(cxs)), 4),
            "cy_mean": round(float(np.mean(cys)), 4),
            "cx_list": [round(v, 4) for v in cxs],
            "cy_list": [round(v, 4) for v in cys],
        }

    return {
        "cx_all": [round(v, 4) for v in cx_all],
        "cy_all": [round(v, 4) for v in cy_all],
        "cx_stats": _stats_dict(cx_all),
        "cy_stats": _stats_dict(cy_all),
        "per_class_centers": per_class_centers,
        "top4_classes": top4_classes,
    }


def cooccurrence_matrix(data_dir: str | Path) -> dict[str, Any]:
    """Build a 10×10 matrix counting class co-occurrences within the same image.

    Entry ``[i][j]`` is the number of annotations of class ``i`` that share an
    image with at least one annotation of class ``j``. The diagonal equals the
    total annotation count for that class.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``matrix``: 10×10 list[list[int]] — raw co-occurrence counts.
        - ``matrix_normalized``: 10×10 list[list[float]] — row-normalised (each
          row divided by its diagonal, i.e. fraction of class-i annotations that
          co-occur with class-j).
        - ``top_pairs``: list of {class_a, class_b, count} for the top 10
          off-diagonal pairs by count.
        - ``class_names``: list[str] — row/column labels.
    """
    data_dir = Path(data_dir)

    matrix: list[list[int]] = [[0] * 10 for _ in range(10)]

    for _, lf in _collect_label_files(data_dir):
        lines = [lbl.strip() for lbl in lf.read_text(encoding="utf-8").splitlines() if lbl.strip()]
        cids = [int(lbl.split()[0]) for lbl in lines if lbl.split()]
        for i in cids:
            for j in cids:
                if 0 <= i < 10 and 0 <= j < 10:
                    matrix[i][j] += 1

    matrix_normalized: list[list[float]] = []
    for i in range(10):
        denom = matrix[i][i] if matrix[i][i] > 0 else 1
        matrix_normalized.append([round(matrix[i][j] / denom, 4) for j in range(10)])

    top_pairs: list[dict[str, Any]] = []
    for i in range(10):
        for j in range(i + 1, 10):
            count = matrix[i][j]
            if count > 0:
                top_pairs.append(
                    {"class_a": NWPU_CLASSES[i], "class_b": NWPU_CLASSES[j], "count": count}
                )
    top_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "matrix": matrix,
        "matrix_normalized": matrix_normalized,
        "top_pairs": top_pairs[:10],
        "class_names": NWPU_CLASSES,
    }


def image_resolution(data_dir: str | Path) -> dict[str, Any]:
    """Sample image dimensions across the dataset using OpenCV.

    Reads up to 100 images per split and records width × height. For the
    Roboflow-exported NWPU dataset all images are resized to 640×640.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``widths``: list[int] of sampled image widths.
        - ``heights``: list[int] of sampled image heights.
        - ``unique_resolutions``: {\"WxH\": count} — all distinct WxH strings seen.
        - ``all_same_resolution``: bool.
        - ``dominant_resolution``: str — most common WxH string.
        - ``width_stats``, ``height_stats``: descriptive stats.
        - ``sample_size``: int.
    """
    data_dir = Path(data_dir)
    widths: list[int] = []
    heights: list[int] = []

    rng = np.random.default_rng(42)

    for split in SPLITS:
        img_dir = data_dir / split / "images"
        all_imgs: list[Path] = []
        for pattern in IMAGE_EXTENSIONS:
            all_imgs.extend(img_dir.glob(pattern))

        k = min(100, len(all_imgs))
        sample = [all_imgs[i] for i in rng.choice(len(all_imgs), size=k, replace=False).tolist()]

        for img_path in sample:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                h, w = frame.shape[:2]
                widths.append(w)
                heights.append(h)

    resolution_counts: Counter[str] = Counter(
        f"{w}x{h}" for w, h in zip(widths, heights)
    )
    dominant = resolution_counts.most_common(1)[0][0] if resolution_counts else "unknown"

    return {
        "widths": widths,
        "heights": heights,
        "unique_resolutions": dict(resolution_counts),
        "all_same_resolution": len(resolution_counts) == 1,
        "dominant_resolution": dominant,
        "width_stats": _stats_dict([float(w) for w in widths]),
        "height_stats": _stats_dict([float(h) for h in heights]),
        "sample_size": len(widths),
    }


def negative_images(data_dir: str | Path) -> dict[str, Any]:
    """Identify images with empty label files (no annotations).

    The original NWPU VHR-10 dataset includes 150 background images with no
    objects. The Roboflow export used here contains only the 650 positive images,
    so this function is expected to return zero negatives. It is retained as a
    reusable check for future dataset versions or augmented splits.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict with keys:

        - ``count``: int — number of images with no annotations.
        - ``per_split``: {split: int} — breakdown by split.
        - ``files``: list[str] — relative paths of empty label files.
        - ``total_label_files``: int — total label files checked.
        - ``note``: str — contextual explanation.
    """
    data_dir = Path(data_dir)
    negatives: list[str] = []
    per_split: dict[str, int] = {}
    total = 0

    for split in SPLITS:
        lbl_dir = data_dir / split / "labels"
        count = 0
        for lf in sorted(lbl_dir.glob("*.txt")):
            total += 1
            content = lf.read_text(encoding="utf-8").strip()
            if not content:
                count += 1
                rel = f"{split}/labels/{lf.name}"
                negatives.append(rel)
        per_split[split] = count

    note = (
        "Zero negative images found. The Roboflow export of NWPU VHR-10 includes "
        "only the 650 positive images from the original dataset; the 150 background "
        "images (negative image set) were excluded during the Roboflow project setup."
        if not negatives
        else f"{len(negatives)} background images found across splits."
    )

    return {
        "count": len(negatives),
        "per_split": per_split,
        "files": negatives,
        "total_label_files": total,
        "note": note,
    }


def run_all_eda(data_dir: str | Path) -> dict[str, Any]:
    """Execute all seven EDA functions and return a combined results dict.

    Args:
        data_dir: Path to the raw dataset root.

    Returns:
        Dict keyed by analysis name, each containing the respective function's
        output. Suitable for direct serialisation to ``dataset_stats.json``.
    """
    data_dir = Path(data_dir)
    return {
        "class_distribution": class_distribution(data_dir),
        "class_imbalance": class_imbalance(data_dir),
        "bbox_analysis": bbox_analysis(data_dir),
        "spatial_heatmap": spatial_heatmap(data_dir),
        "cooccurrence_matrix": cooccurrence_matrix(data_dir),
        "image_resolution": image_resolution(data_dir),
        "negative_images": negative_images(data_dir),
    }
