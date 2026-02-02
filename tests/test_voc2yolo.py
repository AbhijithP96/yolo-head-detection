import pytest

from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

from yolo_head_detection.data_utils.voc2yolo import (
    _convert_box,
    _read_and_convert_xml,
    _create_yaml_file,
)


##-------------------------------------------------------------
# Helper Functions for Tests
##-------------------------------------------------------------
def write_xml(tmp_path: Path, content: str) -> Path:
    file = tmp_path / "test.xml"
    file.write_text(content)
    return file


##-------------------------------------------------------------
# Tests for _convert_box function in voc2yolo.py
##-------------------------------------------------------------
def test_basic_box_conversion():
    """Standard box in center of image should normalize correctly."""
    bbox = [10, 30, 20, 60]  # xmin, xmax, ymin, ymax
    w, h = 100, 100

    class_id, x, y, bw, bh = _convert_box(bbox, w, h)

    assert class_id == 0
    assert x == pytest.approx(0.19)  # center x = 19 / 100
    assert y == pytest.approx(0.39)  # center y = 39 / 100
    assert bw == pytest.approx(0.2)  # width = 20 / 100
    assert bh == pytest.approx(0.4)  # height = 40 / 100


def test_full_image_box():
    """Box covering entire image should return center 0.49,0.49 and size 1,1."""
    bbox = [0, 200, 0, 100]
    w, h = 200, 100

    _, x, y, bw, bh = _convert_box(bbox, w, h)

    assert x == pytest.approx(99 / 200)
    assert y == pytest.approx(49 / 100)
    assert bw == pytest.approx(1.0)
    assert bh == pytest.approx(1.0)


def test_box_touching_edges():
    """Boxes touching borders should still normalize within [0,1]."""
    bbox = [0, 50, 0, 50]
    w, h = 100, 100

    _, x, y, bw, bh = _convert_box(bbox, w, h)

    assert 0 <= x <= 1
    assert 0 <= y <= 1
    assert 0 <= bw <= 1
    assert 0 <= bh <= 1


def test_small_box():
    """Very small box should produce small normalized values."""
    bbox = [10, 11, 20, 21]
    w, h = 100, 100

    _, x, y, bw, bh = _convert_box(bbox, w, h)

    assert bw == pytest.approx(0.01)
    assert bh == pytest.approx(0.01)


def test_non_square_image():
    """Normalization should respect different width/height scales."""
    bbox = [50, 150, 25, 75]
    w, h = 200, 100

    _, x, y, bw, bh = _convert_box(bbox, w, h)

    assert x == pytest.approx(99 / 200)
    assert y == pytest.approx(49 / 100)
    assert bw == pytest.approx(0.5)
    assert bh == pytest.approx(0.5)


##-------------------------------------------------------------
# Tests for _read_and_convert_xml function in voc2yolo.py
##-------------------------------------------------------------


def test_single_object_conversion(tmp_path):
    """Single bounding box should be converted correctly."""
    xml = """<annotation>
        <size>
            <width>100</width>
            <height>200</height>
        </size>
        <object>
            <name>head</name>
            <bndbox>
                <xmin>10</xmin><ymin>20</ymin>
                <xmax>30</xmax><ymax>60</ymax>
            </bndbox>
        </object>
    </annotation>"""

    xml_file = write_xml(tmp_path, xml)
    boxes = _read_and_convert_xml(xml_file)

    assert len(boxes) == 1
    cls, x, y, w, h = boxes[0]

    assert cls == 0
    assert x == pytest.approx(0.19)  # (10+30)/2 -1 / 100
    assert y == pytest.approx(39 / 200)  # (20+60)/2 -1 / 200
    assert w == pytest.approx(0.20)  # (30-10)/100
    assert h == pytest.approx(0.20)  # (60-20)/200


def test_multiple_objects(tmp_path):
    """All objects in XML should be converted."""
    xml = """<annotation>
        <size>
            <width>100</width>
            <height>100</height>
        </size>
        <object>
            <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
        </object>
        <object>
            <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>100</xmax><ymax>100</ymax></bndbox>
        </object>
    </annotation>"""

    xml_file = write_xml(tmp_path, xml)
    boxes = _read_and_convert_xml(xml_file)

    assert len(boxes) == 2
    assert boxes[0][1:] == pytest.approx((0.24, 0.24, 0.5, 0.5))
    assert boxes[1][1:] == pytest.approx((0.74, 0.74, 0.5, 0.5))


def test_no_objects(tmp_path):
    """XML with no <object> tags should return empty list."""
    xml = """<annotation>
        <size>
            <width>100</width>
            <height>100</height>
        </size>
    </annotation>"""

    xml_file = write_xml(tmp_path, xml)
    boxes = _read_and_convert_xml(xml_file)

    assert boxes == []


def test_invalid_xml_raises(tmp_path):
    """Malformed XML should raise an error."""
    xml_file = write_xml(tmp_path, "<annotation><bad></annotation>")

    with pytest.raises(Exception):
        _read_and_convert_xml(xml_file)


def test_missing_size_tag(tmp_path):
    """Missing size info should raise error."""
    xml = """<annotation>
        <object>
            <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox>
        </object>
    </annotation>"""

    xml_file = write_xml(tmp_path, xml)

    with pytest.raises(Exception):
        _read_and_convert_xml(xml_file)


##-------------------------------------------------------------
# Tests for _create_yaml_file function in voc2yolo.py
##-------------------------------------------------------------


def test_create_yaml_file(tmp_path):
    """YAML file should be created with correct content."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    _create_yaml_file(out_dir)

    yaml_file = out_dir / "hollywoodheads.yaml"

    assert yaml_file.exists()

    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    assert content["path"] == str(out_dir)
    assert content["train"] == "images/train"
    assert content["val"] == "images/val"
    assert content["test"] == "images/test"
    assert len(content["names"].keys()) == 1
    assert content["names"][0] == "head"
