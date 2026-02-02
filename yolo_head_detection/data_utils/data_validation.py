from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from loguru import logger


def _check_image_file(image_path: Path, size: tuple) -> bool:
    """
    Check if the image file is valid and matches the specified size.

    This function verifies that the image can be opened, loaded, and has the exact
    dimensions provided in the size tuple.

    Args:
        image_path (Path): Path to the image file to check.
        size (tuple): A tuple of two integers (width, height) representing the expected image size.

    Returns:
        bool: True if the image is valid and matches the size, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.load()

        with Image.open(image_path) as img:
            w_img, h_img = img.size

            if w_img != size[0] and h_img != size[1]:
                raise Exception

        return True

    except:
        return False


def _to_discard_data(filename: str, data_path: Path) -> bool:
    """
    Determine if the data for a given filename should be discarded based on validation checks.

    This function checks the XML annotation file for the image, verifies the image file,
    ensures bounding boxes are valid, and checks minimum size constraints.

    Args:
        filename (str): The base filename (without extension) of the data sample.
        data_path (Path): Path to the directory containing Annotations and JPEGImages subdirectories.

    Returns:
        bool: True if the data should be discarded, False if it should be kept.
    """
    MIN_W, MIN_H = 5, 5

    try:
        xml_path = data_path / "Annotations" / f"{filename}.xml"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_path = data_path / "JPEGImages" / root.findtext("filename")

        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        depth = int(size.findtext("depth"))

        if depth != 3:
            return True

        if not _check_image_file(image_path, (width, height)):
            return True

        objects = root.findall("object")

        if len(objects) == 0:
            return False  # include background images

        for obj in objects:
            bndbox = obj.find("bndbox")

            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))

            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                return True

            if xmax - xmin < MIN_W or ymax - ymin < MIN_H:
                return True

        return False

    except Exception:
        return True


def validate_data(raw_data_dir: Path, interim_data_dir: Path) -> None:
    """
    Validate the dataset by checking for split overlaps and filtering out invalid data samples.

    This function reads the train/val/test splits, ensures no overlaps between them,
    and for each split, validates each data sample using _to_discard_data, keeping only valid ones.
    The filtered lists are written to the interim data directory.

    Args:
        raw_data_dir (Path): Path to the raw data directory containing Splits, Annotations, and JPEGImages.
        interim_data_dir (Path): Path to the interim data directory where filtered split files will be saved.

    Returns:
        None
    """
    logger.info("Validating Dataset")

    data_path = raw_data_dir / "HollywoodHeads"
    split_dir = data_path / "Splits"

    # logger.info('Creating Output directory')
    out_dir = interim_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Checking Train/val/test overlap")
    train = set((split_dir / "train.txt").read_text().splitlines())
    val = set((split_dir / "val.txt").read_text().splitlines())
    test = set((split_dir / "test.txt").read_text().splitlines())

    assert train.isdisjoint(val), "Train/Val overlap detected"
    assert train.isdisjoint(test), "Train/Test overlap detected"
    assert val.isdisjoint(test), "Val/Test overlap detected"

    for split in split_dir.glob("*.txt"):
        logger.info(f"Checkinng split {split.stem}")

        kept = []

        with open(split, "r") as f:
            filenames = f.read().splitlines()

        for filename in tqdm(filenames, desc=split.name):
            if not _to_discard_data(filename, data_path):
                kept.append(filename)

        out_path = out_dir / split.name
        out_path.write_text("\n".join(kept) + "\n")

        logger.info(f"{split.name}: kept {len(kept)} / {len(filenames)}")
