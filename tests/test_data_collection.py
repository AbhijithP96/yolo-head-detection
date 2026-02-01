import pytest

import tempfile
import zipfile
import io
import requests
from pathlib import Path

from yolo_head_detection.data_utils.data_collection import (
    _download_data,
    _extract_zip,
    _create_split,
    _if_dataset_exist,
)


class FakeResponse:
    def __init__(self, content: bytes, status_code: int):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP Error: {self.status_code}")

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def make_test_zip() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("test_file.txt", "This is a test file.")
    return zip_buffer.getvalue()


def make_fake_annotations(dir_path: Path, num_files: int = 10):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_files):
        (dir_path / f"image_{i}.xml").write_text("<annotation></annotation>")


def make_fake_dataset(base: Path, num_files: int = 3, mismatch: bool = False):
    img_dir = base / "HollywoodHeads" / "JPEGImages"
    ann_dir = base / "HollywoodHeads" / "Annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_files):
        (img_dir / f"image_{i}.jpeg").write_text("fake image")
        # optionally create mismatched annotations
        ann_name = f"image_{i + 1}.xml" if mismatch else f"image_{i}.xml"
        (ann_dir / ann_name).write_text("<annotation></annotation>")


def test_download_data_no_http_error(monkeypatch):
    """Test that _download_data successfully downloads and writes file on 200 OK."""
    zip_content = make_test_zip()

    def fake_get_success(url, stream=True):
        return FakeResponse(content=zip_content, status_code=200)

    monkeypatch.setattr(requests, "get", fake_get_success)
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = tmpdir + "/fake.zip"
        _download_data(url="http://fakeurl.com/fake.zip", download_path=Path(dest_path))

        with open(dest_path, "rb") as f:
            content = f.read()
            assert (
                content == zip_content
            ), "Downloaded content does not match expected content."


def test_download_data_http_error(monkeypatch):
    """Test that _download_data raises RuntimeError on HTTP error."""

    def fake_get_failure(url, stream=True):
        return FakeResponse(content=b"", status_code=404)

    monkeypatch.setattr(requests, "get", fake_get_failure)

    with pytest.raises(Exception):
        _download_data(
            url="http://fakeurl.com/fake.zip", download_path="should_not_matter.zip"
        )


def test_extract_zip(tmp_path):
    """Test that _extract_zip extracts files correctly and handles missing files."""
    zip_content = make_test_zip()

    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(zip_content)

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    _extract_zip(zip_path=zip_path, raw_dir=extract_dir)

    extracted = list(extract_dir.iterdir())
    assert len(extracted) == 1
    extracted_file = extracted[0]

    assert extracted_file.name == "test_file.txt"
    assert extracted_file.read_text() == "This is a test file."

    with pytest.raises(Exception):
        _extract_zip(zip_path="non_existant.zip", raw_dir=tmp_path)


@pytest.mark.parametrize("split_exit", [True, False])
def test_create_split(tmp_path, split_exit):
    """Test that _create_split creates/overwrites train/val/test split files correctly."""
    ann_dir = tmp_path / "HollywoodHeads" / "Annotations"
    make_fake_annotations(ann_dir, num_files=40)

    split_dir = ann_dir.parent / "Splits"

    if split_exit:
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train.txt").write_text("fake_train\n")
        (split_dir / "val.txt").write_text("fake_val\n")
        (split_dir / "test.txt").write_text("fake_test\n")

    _create_split(data_dir=tmp_path)

    assert split_dir.exists()
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        file_path = split_dir / split_file
        assert file_path.exists()

        lines = file_path.read_text().strip().splitlines()
        assert len(lines) > 0

        if split_exit:
            assert lines[0].startswith("fake_")

    if not split_exit:
        total_files = sum(
            len((split_dir / f).read_text().strip().splitlines())
            for f in ["train.txt", "val.txt", "test.txt"]
        )
        assert total_files == 40

    with pytest.raises(Exception):
        _create_split(tmp_path / "non_existant_dir")


@pytest.mark.parametrize(
    "setup_func, expected",
    [
        (lambda tmp: tmp / "missing", False),  # dataset folder missing
        (lambda tmp: tmp, False),  # missing HollywoodHeads/JPEGImages/Annotations
        (
            lambda tmp: make_fake_dataset(tmp, num_files=0) or tmp,
            False,
        ),  # empty directories
        (
            lambda tmp: make_fake_dataset(tmp, num_files=3, mismatch=True) or tmp,
            False,
        ),  # mismatch
        (lambda tmp: make_fake_dataset(tmp, num_files=3) or tmp, True),  # valid dataset
    ],
)
def test_if_dataset_exist(tmp_path, setup_func, expected):
    """Test that _if_dataset_exist correctly validates dataset presence and integrity."""
    data_dir = setup_func(tmp_path)
    result = _if_dataset_exist(data_dir)
    assert result == expected
