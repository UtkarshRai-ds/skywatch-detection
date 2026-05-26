"""Model evaluation utilities for YOLOv8 checkpoints on NWPU VHR-10.

Provides functions to load trained ``.pt`` files, extract architecture
metadata, and structure evaluation results (from hardcoded test-set metrics or
from a live ``model.val()`` run) into a consistent dict suitable for JSON
serialisation.

Typical usage::

    from src.models.evaluate import get_model_info, build_test_results

    info = get_model_info(Path("models/yolov8n_best.pt"))
    results = build_test_results(
        model_info=info,
        overall={"mAP50": 0.689, "precision": 0.794, "recall": 0.634, "mAP50_95": 0.426},
        per_class={"airplane": 0.561, ...},
    )
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Canonical class names matching the YOLO index order stored in the checkpoints.
from src.data.loader import NWPU_CLASSES


@dataclass
class ModelInfo:
    """Architecture and filesystem metadata for a trained YOLOv8 checkpoint."""

    model_name: str
    path: str
    params: int
    architecture: str
    task: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_model_info(model_path: str | Path) -> ModelInfo:
    """Load a YOLOv8 checkpoint and return its architecture metadata.

    Args:
        model_path: Path to a ``.pt`` weights file.

    Returns:
        Populated :class:`ModelInfo` dataclass.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    from ultralytics import YOLO  # deferred to avoid slow import at module level

    model = YOLO(str(model_path))
    params = sum(p.numel() for p in model.model.parameters())
    arch = model_path.stem.split("_")[0]  # "yolov8n" from "yolov8n_best"

    return ModelInfo(
        model_name=arch,
        path=str(model_path.resolve()),
        params=params,
        architecture=arch.upper(),
        task=model.task,
    )


def measure_inference_speed(
    model_path: str | Path,
    image_paths: list[Path],
    n_warmup: int = 3,
) -> dict[str, float]:
    """Measure average inference speed on a list of images (CPU).

    Runs ``n_warmup`` warm-up passes before timing to avoid cold-start
    bias from JIT compilation.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        image_paths: List of image paths to use for timing.
        n_warmup: Number of warm-up inference calls before measurement.

    Returns:
        Dict with keys ``preprocess_ms``, ``inference_ms``, ``postprocess_ms``,
        ``total_ms``, and ``fps`` — all averaged over the provided images.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    images = [str(p) for p in image_paths]

    for img in images[:n_warmup]:
        model(img, verbose=False)

    pre_ms, inf_ms, post_ms = [], [], []
    for img in images:
        res = model(img, verbose=False)
        spd = res[0].speed
        pre_ms.append(spd["preprocess"])
        inf_ms.append(spd["inference"])
        post_ms.append(spd["postprocess"])

    avg_pre = float(sum(pre_ms) / len(pre_ms))
    avg_inf = float(sum(inf_ms) / len(inf_ms))
    avg_post = float(sum(post_ms) / len(post_ms))
    total = avg_pre + avg_inf + avg_post

    return {
        "preprocess_ms":  round(avg_pre,  2),
        "inference_ms":   round(avg_inf,  2),
        "postprocess_ms": round(avg_post, 2),
        "total_ms":        round(total,    2),
        "fps":             round(1000 / total, 1) if total > 0 else 0.0,
        "device":          "cpu",
    }


def parse_training_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse a Ultralytics results CSV into a list of per-epoch dicts.

    Strips whitespace from column names and values, converts numerics.

    Args:
        csv_path: Path to the ``results.csv`` produced during training.

    Returns:
        List of dicts, one per epoch, with cleaned column names as keys.
    """
    csv_path = Path(csv_path)
    history: list[dict[str, Any]] = []

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            row: dict[str, Any] = {}
            for k, v in raw_row.items():
                k_clean = k.strip()
                try:
                    row[k_clean] = float(v.strip())
                except (ValueError, AttributeError):
                    row[k_clean] = v.strip() if isinstance(v, str) else v
            history.append(row)

    return history


def build_test_results(
    model_info: ModelInfo,
    overall: dict[str, float],
    per_class: dict[str, float],
    speed: dict[str, float] | None = None,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    """Assemble a structured evaluation results dict for JSON output.

    Args:
        model_info: Architecture metadata from :func:`get_model_info`.
        overall: Dict with keys ``mAP50``, ``precision``, ``recall``,
            ``mAP50_95`` — test-set overall metrics.
        per_class: Mapping of class name → mAP50 on the test set.
            Keys use the canonical NWPU class names (may contain spaces).
        speed: Optional speed benchmark dict from :func:`measure_inference_speed`.
        best_epoch: Epoch index at which the checkpoint was saved.

    Returns:
        Fully structured dict ready for ``json.dumps``.
    """
    # Normalise class names: "storage tank" → "storage_tank" for JSON keys
    per_class_norm: dict[str, float] = {}
    for name, val in per_class.items():
        key = name.replace(" ", "_")
        per_class_norm[key] = round(float(val), 4)

    # Derive class ranking for quick reference
    sorted_classes = sorted(per_class_norm.items(), key=lambda x: x[1], reverse=True)

    return {
        "model_name":   model_info.model_name,
        "architecture": model_info.architecture,
        "params":       model_info.params,
        "task":         model_info.task,
        "best_epoch":   best_epoch,
        "overall": {
            "mAP50":     round(float(overall["mAP50"]),     4),
            "precision": round(float(overall["precision"]), 4),
            "recall":    round(float(overall["recall"]),    4),
            "mAP50_95":  round(float(overall["mAP50_95"]), 4),
        },
        "per_class_mAP50": per_class_norm,
        "class_ranking":   [name for name, _ in sorted_classes],
        "best_class": sorted_classes[0][0] if sorted_classes else None,
        "worst_class": sorted_classes[-1][0] if sorted_classes else None,
        "speed": speed or {},
    }


def run_live_val(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    conf: float = 0.25,
    iou: float = 0.6,
) -> dict[str, Any]:
    """Run a live validation pass with Ultralytics and return raw metrics.

    This re-runs evaluation on the specified split using the model's built-in
    ``val()`` method. Results will differ slightly from the Colab test run due
    to hardware and environment differences; use :func:`build_test_results`
    with hardcoded values for reproducible reporting.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        data_yaml: Path to the dataset ``data.yaml`` file.
        split: Dataset split to evaluate on — ``"test"``, ``"val"``, etc.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        Dict with keys from the Ultralytics ``Results`` object: ``mAP50``,
        ``mAP50_95``, ``precision``, ``recall``, and ``per_class_mAP50``.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    ap50 = val_results.ap_class_index  # class indices

    per_class: dict[str, float] = {}
    if hasattr(val_results, "ap") and val_results.ap is not None:
        for idx, cls_idx in enumerate(ap50):
            name = NWPU_CLASSES[cls_idx].replace(" ", "_")
            per_class[name] = round(float(val_results.ap[idx]), 4)

    return {
        "mAP50":          round(float(val_results.box.map50),  4),
        "mAP50_95":       round(float(val_results.box.map),    4),
        "precision":      round(float(val_results.box.mp),     4),
        "recall":         round(float(val_results.box.mr),     4),
        "per_class_mAP50": per_class,
    }
