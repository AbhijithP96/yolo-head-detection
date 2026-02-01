import pytest

from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

from yolo_head_detection.data_utils.data_validation import _check_image_file, _to_discard_data

##-------------------------------------------------------------
# Helper Functions for Tests
##-------------------------------------------------------------
def create_image(path: Path, size=(100, 100), color=(255, 0, 0)):
    """Helper to create a real image file."""
    img = Image.new("RGB", size, color)
    img.save(path)

def create_xml(path: Path, filename: str, width=100, height=100, depth=3, boxes=None):
    """
    Create a Pascal VOC style XML annotation file.
    boxes = list of (xmin, ymin, xmax, ymax)
    """
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "filename").text = f"{filename}.jpeg"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    obj = ET.SubElement(annotation, "object")
    for (xmin, ymin, xmax, ymax) in boxes:
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(path)


def setup_dataset(tmp_path):
    data_path = tmp_path
    (data_path / "Annotations").mkdir()
    (data_path / "JPEGImages").mkdir()
    return data_path

##-------------------------------------------------------------
# Tests for check_image_file function in data_validation.py
##-------------------------------------------------------------


def test_valid_image_correct_size(tmp_path):
    """Return True for a valid image matching expected size."""
    img_path = tmp_path / "valid.jpg"
    create_image(img_path, size=(64, 64))

    assert _check_image_file(img_path, (64, 64)) is True


def test_valid_image_wrong_size(tmp_path):
    """Return False when image dimensions do not match expected size."""
    img_path = tmp_path / "wrong_size.jpg"
    create_image(img_path, size=(32, 32))

    assert _check_image_file(img_path, (64, 64)) is False


def test_corrupted_image(tmp_path):
    """Return False for a corrupted or non-image file."""
    img_path = tmp_path / "corrupt.jpg"
    img_path.write_bytes(b"this is not a real image")

    assert _check_image_file(img_path, (64, 64)) is False


def test_missing_image_file(tmp_path):
    """Return False when the image file does not exist."""
    img_path = tmp_path / "missing.jpg"

    assert _check_image_file(img_path, (64, 64)) is False


def test_valid_image_different_format(tmp_path):
    """Return True for valid PNG image matching expected size."""
    img_path = tmp_path / "image.png"
    create_image(img_path, size=(128, 128))

    assert _check_image_file(img_path, (128, 128)) is True

##-------------------------------------------------------------
# Tests for _to_discard_data function in data_validation.py
##-------------------------------------------------------------

def test_valid_sample_kept(tmp_path):
    """Valid image + valid bbox should NOT be discarded."""
    data_path = setup_dataset(tmp_path)
    create_image(data_path / "JPEGImages" / "img1.jpeg", (100, 100))
    create_xml(
        data_path / "Annotations" / "img1.xml",
        "img1",
        boxes=[(10, 10, 50, 50)]
    )

    assert _to_discard_data("img1", data_path) is False


def test_discard_if_image_missing(tmp_path):
    """Missing image file should be discarded."""
    data_path = setup_dataset(tmp_path)
    create_xml(data_path / "Annotations" / "img1.xml", "img1", boxes=[(10, 10, 50, 50)])

    assert _to_discard_data("img1", data_path) is True


def test_discard_if_wrong_depth(tmp_path):
    """Non-RGB images (depth != 3) should be discarded."""
    data_path = setup_dataset(tmp_path)
    create_image(data_path / "JPEGImages" / "img1.jpeg", (100, 100))
    create_xml(
        data_path / "Annotations" / "img1.xml",
        "img1",
        depth=1,
        boxes=[(10, 10, 50, 50)]
    )

    assert _to_discard_data("img1", data_path) is True


def test_discard_if_size_mismatch(tmp_path):
    """Image size not matching XML should be discarded."""
    data_path = setup_dataset(tmp_path)
    create_image(data_path / "JPEGImages" / "img1.jpeg", (200, 200))  # actual size
    create_xml(
        data_path / "Annotations" / "img1.xml",
        "img1",
        width=100,
        height=100,
        boxes=[(10, 10, 50, 50)]
    )

    assert _to_discard_data("img1", data_path) is True


def test_discard_if_bbox_out_of_bounds(tmp_path):
    """Bounding box outside image should be discarded."""
    data_path = setup_dataset(tmp_path)
    create_image(data_path / "JPEGImages" / "img1.jpeg", (100, 100))
    create_xml(
        data_path / "Annotations" / "img1.xml",
        "img1",
        boxes=[(10, 10, 150, 50)]  # xmax > width
    )

    assert _to_discard_data("img1", data_path) is True


def test_discard_if_bbox_too_small(tmp_path):
    """Bounding box smaller than minimum size should be discarded."""
    data_path = setup_dataset(tmp_path)
    create_image(data_path / "JPEGImages" / "img1.jpeg", (100, 100))
    create_xml(
        data_path / "Annotations" / "img1.xml",
        "img1",
        boxes=[(10, 10, 15, 15)]  # 5x5 < MIN_W/H=10
    )

    assert _to_discard_data("img1", data_path) is True


def test_discard_if_xml_corrupted(tmp_path):
    """Corrupted XML should cause discard."""
    data_path = setup_dataset(tmp_path)
    create_image(data_path / "JPEGImages" / "img1.jpeg", (100, 100))

    xml_path = data_path / "Annotations" / "img1.xml"
    xml_path.write_text("<not valid xml")

    assert _to_discard_data("img1", data_path) is True