"""Tests for src/data/converter.py — Pascal VOC XML to YOLO format conversion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.data.converter import (
    BoundingBox,
    VocToYoloConverter,
    _parse_voc_xml,
    _voc_bbox_to_yolo,
)
from src.data.loader import NWPU_CLASSES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_voc_xml(
    path: Path,
    filename: str,
    width: int,
    height: int,
    objects: list[tuple[str, int, int, int, int]],
) -> None:
    """Write a minimal Pascal VOC XML file to *path*."""
    lines = [
        '<?xml version="1.0"?>',
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "    <depth>3</depth>",
        "  </size>",
    ]
    for name, xmin, ymin, xmax, ymax in objects:
        lines += [
            "  <object>",
            f"    <name>{name}</name>",
            "    <bndbox>",
            f"      <xmin>{xmin}</xmin>",
            f"      <ymin>{ymin}</ymin>",
            f"      <xmax>{xmax}</xmax>",
            f"      <ymax>{ymax}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    lines.append("</annotation>")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── _voc_bbox_to_yolo ──────────────────────────────────────────────────────────

def test_voc_bbox_to_yolo_all_values_within_unit_range() -> None:
    bbox = BoundingBox(100, 200, 300, 400)
    result = _voc_bbox_to_yolo(bbox, img_width=640, img_height=640)
    assert result is not None
    for val in result:
        assert 0.0 <= val <= 1.0


def test_voc_bbox_to_yolo_correct_normalisation() -> None:
    # Centred box: xmin=0, ymin=0, xmax=640, ymax=640 on 640×640 image
    bbox = BoundingBox(0, 0, 640, 640)
    result = _voc_bbox_to_yolo(bbox, 640, 640)
    assert result is not None
    cx, cy, w, h = result
    assert pytest.approx(cx, abs=1e-6) == 0.5
    assert pytest.approx(cy, abs=1e-6) == 0.5
    assert pytest.approx(w,  abs=1e-6) == 1.0
    assert pytest.approx(h,  abs=1e-6) == 1.0


def test_voc_bbox_to_yolo_returns_none_for_degenerate_box() -> None:
    # xmax == xmin — zero-area box
    assert _voc_bbox_to_yolo(BoundingBox(100, 100, 100, 200), 640, 640) is None
    # ymax == ymin
    assert _voc_bbox_to_yolo(BoundingBox(100, 200, 200, 200), 640, 640) is None


def test_voc_bbox_to_yolo_clamps_minor_overflow() -> None:
    # Slightly over image boundary due to rounding; should clamp, not return None
    bbox = BoundingBox(0, 0, 641, 641)
    result = _voc_bbox_to_yolo(bbox, 640, 640)
    assert result is not None
    for val in result:
        assert 0.0 <= val <= 1.0


# ── _parse_voc_xml ─────────────────────────────────────────────────────────────

def test_parse_voc_xml_reads_filename_and_size(tmp_path: Path) -> None:
    xml_path = tmp_path / "img.xml"
    _write_voc_xml(xml_path, "img.jpg", 640, 480, [("airplane", 10, 20, 100, 80)])
    ann = _parse_voc_xml(xml_path)

    assert ann.filename == "img.jpg"
    assert ann.width == 640
    assert ann.height == 480


def test_parse_voc_xml_reads_all_objects(tmp_path: Path) -> None:
    xml_path = tmp_path / "multi.xml"
    _write_voc_xml(
        xml_path, "multi.jpg", 640, 640,
        [("airplane", 10, 10, 100, 100), ("ship", 200, 200, 400, 400)],
    )
    ann = _parse_voc_xml(xml_path)

    assert len(ann.objects) == 2
    names = [name for name, _ in ann.objects]
    assert "airplane" in names
    assert "ship" in names


def test_parse_voc_xml_raises_on_missing_size(tmp_path: Path) -> None:
    xml_path = tmp_path / "no_size.xml"
    xml_path.write_text(
        "<annotation><filename>x.jpg</filename></annotation>", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="<size>"):
        _parse_voc_xml(xml_path)


# ── VocToYoloConverter ─────────────────────────────────────────────────────────

def test_convert_single_produces_yolo_format(tmp_path: Path) -> None:
    xml_path = tmp_path / "img001.xml"
    out_dir = tmp_path / "labels"
    _write_voc_xml(xml_path, "img001.jpg", 640, 640, [("airplane", 100, 100, 300, 200)])

    converter = VocToYoloConverter(NWPU_CLASSES)
    success, _ = converter.convert_single(xml_path, out_dir)

    assert success is True
    label_file = out_dir / "img001.txt"
    assert label_file.exists()

    lines = label_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parts = lines[0].split()
    assert len(parts) == 5                          # class_id cx cy w h
    assert parts[0] == "0"                          # airplane → class 0
    for p in parts[1:]:
        assert 0.0 <= float(p) <= 1.0              # normalised coords


def test_class_index_mapping_is_correct(tmp_path: Path) -> None:
    xml_path = tmp_path / "multi.xml"
    out_dir = tmp_path / "labels"
    _write_voc_xml(
        xml_path, "multi.jpg", 640, 640,
        [
            ("airplane", 10,  10,  100, 100),   # → 0
            ("ship",     200, 200, 400, 400),   # → 1
            ("vehicle",  500, 500, 620, 620),   # → 9
        ],
    )

    converter = VocToYoloConverter(NWPU_CLASSES)
    converter.convert_single(xml_path, out_dir)

    lines = (out_dir / "multi.txt").read_text(encoding="utf-8").strip().splitlines()
    class_ids = {int(line.split()[0]) for line in lines}
    assert 0 in class_ids   # airplane
    assert 1 in class_ids   # ship
    assert 9 in class_ids   # vehicle


def test_unknown_class_is_skipped(tmp_path: Path) -> None:
    xml_path = tmp_path / "unk.xml"
    out_dir = tmp_path / "labels"
    _write_voc_xml(
        xml_path, "unk.jpg", 640, 640,
        [("unknown_object", 10, 10, 100, 100), ("airplane", 200, 200, 400, 400)],
    )

    converter = VocToYoloConverter(NWPU_CLASSES)
    success, _ = converter.convert_single(xml_path, out_dir)

    assert success is True
    lines = (out_dir / "unk.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1              # only airplane written
    assert lines[0].split()[0] == "0"  # class 0


def test_malformed_xml_handled_gracefully(tmp_path: Path) -> None:
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text("this is not xml {{ garbage", encoding="utf-8")

    converter = VocToYoloConverter(NWPU_CLASSES)
    success, message = converter.convert_single(xml_path, tmp_path / "labels")

    assert success is False
    assert "Parse error" in message


def test_convert_directory_counts_converted_files(tmp_path: Path) -> None:
    xml_dir = tmp_path / "xml"
    out_dir = tmp_path / "labels"
    xml_dir.mkdir()

    for i in range(3):
        _write_voc_xml(
            xml_dir / f"img{i:03d}.xml",
            f"img{i:03d}.jpg", 640, 640,
            [("airplane", 10, 10, 100, 100)],
        )

    converter = VocToYoloConverter(NWPU_CLASSES)
    stats = converter.convert_directory(xml_dir, out_dir)

    assert stats.converted == 3
    assert len(stats.errors) == 0
