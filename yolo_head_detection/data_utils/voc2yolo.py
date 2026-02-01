import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Tuple, List
import yaml


def _convert_box(bbox, w, h):
    """
    Convert bounding box coordinates from VOC format to YOLO normalized format.

    Args:
        bbox (list): List of [xmin, xmax, ymin, ymax] in pixel coordinates.
        w (int): Image width in pixels.
        h (int): Image height in pixels.

    Returns:
        tuple: (class_id, x_center, y_center, width, height) normalized to [0,1].
    """
    dw, dh = 1.0 / w, 1.0 / h
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    bw = bbox[1] - bbox[0]
    bh = bbox[3] - bbox[2]

    return (
        0,
        x * dw,
        y * dh,
        bw * dw,
        bh * dh,
    )  # normalized box co-ordiante with single class index


def _read_and_convert_xml(file: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a VOC XML annotation file and convert all bounding boxes to YOLO format.

    Args:
        file (Path): Path to the XML annotation file.

    Returns:
        List[Tuple[int, float, float, float, float]]: List of bounding boxes in YOLO format
            (class_id, x_center, y_center, width, height), all normalized.
    """
    tree = ET.parse(file)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = root.findall("object")
    bbox_list = []

    for obj in objects:
        bndbox = obj.find("bndbox")

        xmin = int(bndbox.findtext("xmin"))
        ymin = int(bndbox.findtext("ymin"))
        xmax = int(bndbox.findtext("xmax"))
        ymax = int(bndbox.findtext("ymax"))

        bbox_yolo = _convert_box([xmin, xmax, ymin, ymax], width, height)
        bbox_list.append(bbox_yolo)

    return bbox_list


def _create_yaml_file(out_dir: Path) -> None:
    """
    Create a YAML configuration file for YOLO training.

    Args:
        out_dir (Path): Directory where the YAML file will be created.
    """
    data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "head"},
    }

    yaml_file = out_dir / "hollywoodheads.yaml"
    with open(yaml_file, "w") as f:
        yaml.safe_dump(data, f)

    logger.info(f"Created Yolo yaml file at: {str(yaml_file)}")


def convert_voc_to_yolo(
    raw_data_dir: Path, interim_data_dir: Path, processed_data_dir: Path
) -> None:
    """
    Convert VOC format dataset to YOLO format.

    Processes XML annotations and images from the raw data directory, converts them
    to YOLO format, and saves them in the processed data directory. Also creates
    a YAML configuration file for YOLO training.

    Args:
        raw_data_dir (Path): Directory containing raw VOC format data.
        interim_data_dir (Path): Directory with interim split files (train.txt, val.txt, etc.).
        processed_data_dir (Path): Directory where processed YOLO data will be saved.

    Raises:
        RuntimeError: If an error occurs during conversion.
    """
    try:
        logger.info("Converting VOC Format to YOLO Format")

        # input directories
        ann_dir_in = raw_data_dir / "HollywoodHeads" / "Annotations"
        img_dir_in = raw_data_dir / "HollywoodHeads" / "JPEGImages"

        # output directories
        img_dir_out = processed_data_dir / "images"
        label_dir_out = processed_data_dir / "labels"

        img_dir_out.mkdir(parents=True, exist_ok=True)
        label_dir_out.mkdir(parents=True, exist_ok=True)

        for split in interim_data_dir.iterdir():
            logger.info(f"Converting {split.stem} set")
            split_dir_img_out = img_dir_out / split.stem
            split_dir_label_out = label_dir_out / split.stem

            split_dir_img_out.mkdir(parents=True, exist_ok=True)
            split_dir_label_out.mkdir(parents=True, exist_ok=True)

            filenames = split.read_text().splitlines()

            for file in tqdm(filenames, desc="Conversion Status"):
                ann_file = ann_dir_in / f"{file}.xml"
                img_file = img_dir_in / f"{file}.jpeg"

                bbox_list = _read_and_convert_xml(ann_file)

                out_label_file = split_dir_label_out / f"{file}.txt"
                label_lines = [" ".join(map(str, bbox)) for bbox in bbox_list]
                out_label_file.write_text("\n".join(label_lines) + "\n")

                out_img_file = split_dir_img_out / f"{file}.jpeg"
                shutil.copy(str(img_file), str(out_img_file))

        _create_yaml_file(processed_data_dir)
        logger.info("Conversion Completed")
    except Exception as e:
        logger.error(f"Error Ocuured during conversion: {str(e)}")
        raise RuntimeError("Conversion Stopped Abruptly!!") from e
