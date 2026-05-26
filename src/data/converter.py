"""Pascal VOC XML to YOLO TXT format converter.

Converts annotations exported in the Pascal VOC XML format (as produced by
LabelImg, CVAT, or the original NWPU VHR-10 release) into the YOLO label
format expected by Ultralytics YOLOv8.

Pascal VOC annotation per object:
    <object>
        <name>airplane</name>
        <bndbox>
            <xmin>100</xmin>  <!-- top-left x, absolute pixel -->
            <ymin>200</ymin>  <!-- top-left y, absolute pixel -->
            <xmax>300</xmax>  <!-- bottom-right x, absolute pixel -->
            <ymax>400</ymax>  <!-- bottom-right y, absolute pixel -->
        </bndbox>
    </object>

YOLO label format (one line per object):
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

where all four coordinates are normalised by the image width/height and
represent the centre point and dimensions of the bounding box.

Usage::

    from src.data.converter import VocToYoloConverter

    converter = VocToYoloConverter(class_names=NWPU_CLASSES)
    stats = converter.convert_directory(
        xml_dir=Path("data/raw_voc/annotations"),
        output_dir=Path("data/raw/train/labels"),
        image_dir=Path("data/raw_voc/images"),  # optional: for dimension lookup
    )
    print(stats)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class BoundingBox(NamedTuple):
    """Absolute pixel bounding box in Pascal VOC layout."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    """Parsed representation of a single Pascal VOC XML file."""

    filename: str
    width: int
    height: int
    objects: list[tuple[str, BoundingBox]] = field(default_factory=list)


@dataclass
class ConversionStats:
    """Aggregated results of a batch conversion."""

    converted: int = 0
    skipped_unknown_class: int = 0
    skipped_invalid_bbox: int = 0
    skipped_missing_size: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Converted: {self.converted} | "
            f"Skipped (unknown class): {self.skipped_unknown_class} | "
            f"Skipped (invalid bbox): {self.skipped_invalid_bbox} | "
            f"Skipped (missing size): {self.skipped_missing_size} | "
            f"Errors: {len(self.errors)}"
        )


def _parse_voc_xml(xml_path: Path) -> VocAnnotation:
    """Parse a Pascal VOC XML annotation file.

    Args:
        xml_path: Path to the ``.xml`` annotation file.

    Returns:
        Populated :class:`VocAnnotation` instance.

    Raises:
        ValueError: If the XML is missing required ``<size>`` fields.
        ET.ParseError: If the file cannot be parsed as XML.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_el = root.find("filename")
    filename = filename_el.text.strip() if filename_el is not None else xml_path.stem

    size_el = root.find("size")
    if size_el is None:
        raise ValueError(f"<size> element missing in {xml_path}")

    width_el = size_el.find("width")
    height_el = size_el.find("height")
    if width_el is None or height_el is None:
        raise ValueError(f"<width> or <height> missing in {xml_path}")

    width = int(width_el.text)
    height = int(height_el.text)

    objects: list[tuple[str, BoundingBox]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bndbox_el = obj.find("bndbox")

        if name_el is None or bndbox_el is None:
            continue

        class_name = name_el.text.strip().lower()
        xmin = float(bndbox_el.findtext("xmin", default="0"))
        ymin = float(bndbox_el.findtext("ymin", default="0"))
        xmax = float(bndbox_el.findtext("xmax", default="0"))
        ymax = float(bndbox_el.findtext("ymax", default="0"))
        objects.append((class_name, BoundingBox(xmin, ymin, xmax, ymax)))

    return VocAnnotation(filename=filename, width=width, height=height, objects=objects)


def _voc_bbox_to_yolo(
    bbox: BoundingBox, img_width: int, img_height: int
) -> tuple[float, float, float, float] | None:
    """Convert an absolute Pascal VOC bounding box to normalised YOLO format.

    Args:
        bbox: Absolute pixel coordinates ``(xmin, ymin, xmax, ymax)``.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple ``(cx_norm, cy_norm, w_norm, h_norm)`` where all values are in
        ``(0, 1]``, or ``None`` if the bounding box is degenerate.
    """
    if bbox.xmax <= bbox.xmin or bbox.ymax <= bbox.ymin:
        return None
    if img_width <= 0 or img_height <= 0:
        return None

    cx = (bbox.xmin + bbox.xmax) / 2.0 / img_width
    cy = (bbox.ymin + bbox.ymax) / 2.0 / img_height
    w = (bbox.xmax - bbox.xmin) / img_width
    h = (bbox.ymax - bbox.ymin) / img_height

    # Clamp to [0, 1] to tolerate minor annotation rounding at image borders.
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    if w <= 0 or h <= 0:
        return None

    return cx, cy, w, h


class VocToYoloConverter:
    """Converts Pascal VOC XML annotations to YOLO ``.txt`` label files.

    Attributes:
        class_names: Ordered list of class names; the list index is the YOLO
            class ID. Names are matched case-insensitively.

    Example::

        from src.data.loader import NWPU_CLASSES
        from src.data.converter import VocToYoloConverter

        converter = VocToYoloConverter(class_names=NWPU_CLASSES)
        stats = converter.convert_directory(
            xml_dir=Path("annotations"),
            output_dir=Path("labels"),
        )
    """

    def __init__(self, class_names: list[str]) -> None:
        """Initialise the converter with an ordered list of class names.

        Args:
            class_names: List of class labels in YOLO index order.
        """
        self.class_names = class_names
        self._class_map: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(class_names)
        }

    def convert_single(
        self, xml_path: Path, output_dir: Path
    ) -> tuple[bool, str]:
        """Convert one Pascal VOC XML file to a YOLO label ``.txt`` file.

        The output file is named after the XML stem (e.g. ``001.xml`` →
        ``001.txt``) and written to ``output_dir``.

        Args:
            xml_path: Path to the source ``.xml`` annotation file.
            output_dir: Directory where the ``.txt`` file will be written.

        Returns:
            Tuple ``(success, message)`` where ``success`` is True on a clean
            conversion and ``message`` describes any issues encountered.
        """
        try:
            annotation = _parse_voc_xml(xml_path)
        except Exception as exc:
            return False, f"Parse error: {exc}"

        lines: list[str] = []
        skipped = 0

        for class_name, bbox in annotation.objects:
            class_id = self._class_map.get(class_name)
            if class_id is None:
                logger.warning("Unknown class '%s' in %s — skipping object.", class_name, xml_path.name)
                skipped += 1
                continue

            yolo_coords = _voc_bbox_to_yolo(bbox, annotation.width, annotation.height)
            if yolo_coords is None:
                logger.warning("Degenerate bbox in %s — skipping object.", xml_path.name)
                skipped += 1
                continue

            cx, cy, w, h = yolo_coords
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{xml_path.stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        msg = f"Wrote {len(lines)} objects"
        if skipped:
            msg += f" ({skipped} skipped)"
        return True, msg

    def convert_directory(
        self,
        xml_dir: Path,
        output_dir: Path,
        recursive: bool = False,
    ) -> ConversionStats:
        """Convert all ``.xml`` files in a directory to YOLO ``.txt`` labels.

        Args:
            xml_dir: Directory containing Pascal VOC ``.xml`` files.
            output_dir: Directory where YOLO ``.txt`` files will be written.
            recursive: If True, recurse into subdirectories of ``xml_dir``.

        Returns:
            :class:`ConversionStats` summarising the batch result.
        """
        pattern = "**/*.xml" if recursive else "*.xml"
        xml_files = sorted(xml_dir.glob(pattern))

        if not xml_files:
            logger.warning("No .xml files found in %s", xml_dir)

        stats = ConversionStats()

        for xml_path in xml_files:
            success, message = self.convert_single(xml_path, output_dir)
            if success:
                stats.converted += 1
            else:
                stats.errors.append(f"{xml_path.name}: {message}")
                logger.error("Failed to convert %s: %s", xml_path.name, message)

        logger.info("Conversion complete. %s", stats)
        return stats
