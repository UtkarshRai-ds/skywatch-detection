"""Reproducible training script for YOLOv8n and YOLOv8s on NWPU VHR-10.

This module documents and re-creates the exact training runs executed in
Google Colab (T4 GPU). It is **not intended to be run locally** — it requires
a GPU and the full dataset accessible at the paths in ``data.yaml``.

To replicate the Colab training:
  1. Open ``notebooks/training.ipynb`` in Colab.
  2. Follow the setup cells (install deps, W&B login, Roboflow download).
  3. Run the train cells — they call the functions defined here.

Colab environment that produced the saved checkpoints:
  - Hardware:     Google Colab T4 GPU (16 GB VRAM)
  - Python:       3.10
  - Ultralytics:  8.2.x
  - CUDA:         12.x
  - Training time: YOLOv8n ≈ 9.4 min, YOLOv8s ≈ 11.4 min (50 epochs each)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Hyperparameters used in the Colab runs ─────────────────────────────────────

TRAIN_CONFIG: dict[str, Any] = {
    "epochs":       50,
    "imgsz":        640,
    "batch":        16,
    "optimizer":    "AdamW",
    "lr0":          0.002,
    "lrf":          0.01,
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box":          7.5,
    "cls":          0.5,
    "dfl":          1.5,
    "hsv_h":        0.015,
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "degrees":      0.0,
    "translate":    0.1,
    "scale":        0.5,
    "shear":        0.0,
    "perspective":  0.0,
    "flipud":       0.0,
    "fliplr":       0.5,
    "mosaic":       1.0,
    "mixup":        0.0,
    "copy_paste":   0.0,
    "workers":      2,
    "seed":         42,
    "project":      "skywatch",
    "exist_ok":     True,
    "pretrained":   True,
    "verbose":      True,
}

MODELS: dict[str, str] = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
}


def fix_data_yaml(yaml_path: str | Path) -> Path:
    """Overwrite garbled Roboflow class names with canonical NWPU VHR-10 names.

    The Roboflow export embeds citation paragraphs in the ``names`` field.
    This function rewrites only the ``names`` key while preserving split paths.
    Must be called once after the Roboflow download, before training.

    Args:
        yaml_path: Path to the ``data.yaml`` produced by Roboflow download.

    Returns:
        Same path with corrected content written in-place.
    """
    import yaml
    from src.data.loader import NWPU_CLASSES

    yaml_path = Path(yaml_path)
    with yaml_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["names"] = NWPU_CLASSES
    cfg["nc"] = len(NWPU_CLASSES)

    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    logger.info("data.yaml class names corrected: %s", yaml_path)
    return yaml_path


def train_model(
    model_name: str,
    data_yaml: str | Path,
    output_dir: str | Path = "runs",
    wandb_project: str | None = "skywatch-detection",
    **overrides: Any,
) -> Path:
    """Train a YOLOv8 model and return the path to the best checkpoint.

    Merges ``TRAIN_CONFIG`` with any ``overrides`` supplied by the caller.
    If ``wandb_project`` is set and W&B is installed, logs all metrics.

    Args:
        model_name: One of the keys in ``MODELS`` — ``"yolov8n"`` or ``"yolov8s"``.
        data_yaml: Path to the fixed ``data.yaml``.
        output_dir: Root directory where Ultralytics saves run artefacts.
        wandb_project: W&B project name. Pass ``None`` to disable W&B logging.
        **overrides: Any :data:`TRAIN_CONFIG` key can be overridden here.

    Returns:
        Path to the ``best.pt`` checkpoint saved by Ultralytics.

    Raises:
        ValueError: If ``model_name`` is not in :data:`MODELS`.
        RuntimeError: If training fails or the checkpoint is not found.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODELS)}")

    from ultralytics import YOLO

    weights = MODELS[model_name]
    cfg = {**TRAIN_CONFIG, **overrides}
    cfg["name"] = model_name
    cfg["data"] = str(data_yaml)

    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=model_name, config=cfg)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")

    logger.info(
        "Starting training: %s  epochs=%d  batch=%d", model_name, cfg["epochs"], cfg["batch"]
    )
    model = YOLO(weights)
    model.train(**cfg, project=str(output_dir))

    # Locate best checkpoint
    best_pt = Path(output_dir) / model_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Expected checkpoint not found: {best_pt}")

    logger.info("Training complete. Best checkpoint: %s", best_pt)
    return best_pt


def train_all(
    data_yaml: str | Path,
    output_dir: str | Path = "runs",
    wandb_project: str | None = "skywatch-detection",
) -> dict[str, Path]:
    """Train both YOLOv8n and YOLOv8s sequentially.

    Args:
        data_yaml: Path to the fixed ``data.yaml``.
        output_dir: Ultralytics output root directory.
        wandb_project: W&B project name; ``None`` disables W&B.

    Returns:
        Dict mapping model name → path to ``best.pt``.
    """
    checkpoints: dict[str, Path] = {}
    for name in MODELS:
        best = train_model(
            model_name=name,
            data_yaml=data_yaml,
            output_dir=output_dir,
            wandb_project=wandb_project,
        )
        checkpoints[name] = best
    return checkpoints
