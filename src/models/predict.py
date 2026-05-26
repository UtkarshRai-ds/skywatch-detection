"""Batch inference on test images with annotated output saving.

Runs YOLOv8 inference on a set of test images, draws bounding boxes with
class labels and confidence scores using OpenCV, and writes annotated copies
to an output directory.

Typical usage::

    from src.models.predict import run_predictions

    detections = run_predictions(
        model_path=Path("models/yolov8s_best.pt"),
        image_dir=Path("data/raw/test/images"),
        output_dir=Path("data/sample_detections"),
        n_images=15,
        conf=0.25,
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.loader import NWPU_CLASSES

# BGR colour palette — one colour per NWPU class, high-contrast on satellite imagery.
_CLASS_COLORS_BGR: list[tuple[int, int, int]] = [
    (0,   165, 255),   # 0  airplane        — orange
    (0,   255, 255),   # 1  ship            — yellow
    (255, 100,   0),   # 2  storage tank    — blue-ish
    (0,   255,   0),   # 3  baseball diamond — green
    (255,   0, 255),   # 4  tennis court    — magenta
    (0,   200, 200),   # 5  basketball court — teal
    (200, 200,   0),   # 6  ground track field — cyan-ish
    (255,   0,   0),   # 7  harbor          — blue
    (50,  205,  50),   # 8  bridge          — lime green
    (0,   100, 255),   # 9  vehicle         — deep orange
]


def _draw_detections(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    class_ids: list[int],
    confidences: list[float],
    font_scale: float = 0.45,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels onto an image array (in-place copy).

    Args:
        image: BGR image array.
        boxes: List of (x1, y1, x2, y2) pixel coordinates.
        class_ids: Class index for each box.
        confidences: Confidence score for each box.
        font_scale: OpenCV font scale for label text.
        thickness: Box line thickness in pixels.

    Returns:
        New array with annotations drawn.
    """
    canvas = image.copy()
    for (x1, y1, x2, y2), cid, conf in zip(boxes, class_ids, confidences):
        color = _CLASS_COLORS_BGR[cid % len(_CLASS_COLORS_BGR)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"{NWPU_CLASSES[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(canvas, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(
            canvas, label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return canvas


def run_predictions(
    model_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    n_images: int = 15,
    conf: float = 0.25,
    iou: float = 0.6,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run inference on a random sample of test images and save annotated results.

    Selects ``n_images`` images at random (fixed seed for reproducibility),
    runs the YOLOv8 model, draws bounding boxes with class labels, and writes
    annotated JPEGs to ``output_dir``.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        image_dir: Directory containing test images (``.jpg`` / ``.png``).
        output_dir: Directory where annotated images will be written.
            Created automatically if absent.
        n_images: Number of test images to process.
        conf: Confidence threshold for detections.
        iou: IoU threshold for NMS.
        seed: Random seed for reproducible image selection.

    Returns:
        List of per-image dicts with keys:

        - ``image_name``: source filename.
        - ``output_path``: path of the saved annotated image.
        - ``n_detections``: total boxes detected.
        - ``classes_detected``: sorted list of unique class names found.
        - ``confidences``: list of per-detection confidence scores.
        - ``speed_ms``: inference time in ms for this image.
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[Path] = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rng = random.Random(seed)
    selected = rng.sample(all_images, min(n_images, len(all_images)))

    model = YOLO(str(model_path))
    model_stem = model_path.stem  # e.g. "yolov8s_best"

    results_log: list[dict[str, Any]] = []

    for img_path in selected:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        res = model(str(img_path), conf=conf, iou=iou, verbose=False)[0]
        spd_ms = round(res.speed["inference"], 2)

        boxes_xyxy: list[tuple[int, int, int, int]] = []
        class_ids: list[int] = []
        confs: list[float] = []

        if res.boxes is not None and len(res.boxes):
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_xyxy.append((int(x1), int(y1), int(x2), int(y2)))
                class_ids.append(int(box.cls[0].item()))
                confs.append(round(float(box.conf[0].item()), 3))

        annotated = _draw_detections(frame, boxes_xyxy, class_ids, confs)

        # Stamp model name and detection count on the image
        stamp = f"{model_stem}  |  {len(boxes_xyxy)} detections  |  conf>={conf}"
        cv2.putText(
            annotated, stamp, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        out_name = f"{model_stem}_{img_path.name}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), annotated)

        unique_classes = sorted({NWPU_CLASSES[cid] for cid in class_ids})
        results_log.append(
            {
                "image_name":       img_path.name,
                "output_path":      str(out_path),
                "n_detections":     len(boxes_xyxy),
                "classes_detected": unique_classes,
                "confidences":      confs,
                "speed_ms":         spd_ms,
            }
        )

    return results_log


def summarise_predictions(results_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image prediction logs into a summary dict.

    Args:
        results_log: Output of :func:`run_predictions`.

    Returns:
        Dict with ``total_detections``, ``avg_detections_per_image``,
        ``class_frequency`` (how often each class appears), ``avg_confidence``,
        and ``images_with_no_detections`` count.
    """
    from collections import Counter

    total_dets = sum(r["n_detections"] for r in results_log)
    all_classes: list[str] = []
    all_confs: list[float] = []
    no_det = 0

    for r in results_log:
        all_classes.extend(r["classes_detected"])
        all_confs.extend(r["confidences"])
        if r["n_detections"] == 0:
            no_det += 1

    n = len(results_log)
    return {
        "images_processed": n,
        "total_detections": total_dets,
        "avg_detections_per_image": round(total_dets / n, 2) if n else 0.0,
        "images_with_no_detections": no_det,
        "class_frequency": dict(Counter(all_classes)),
        "avg_confidence": round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.0,
        "min_confidence": round(min(all_confs), 3) if all_confs else 0.0,
        "max_confidence": round(max(all_confs), 3) if all_confs else 0.0,
    }
