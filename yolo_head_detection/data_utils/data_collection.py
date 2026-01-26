import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from loguru import logger
import os
import time

def _extract(z, m, out_dir):
    target = out_dir / m.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with z.open(m) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)  # 1MB buffer

def _download_data(url: str, download_path: Path):
    """
    Download the dataset archive if not already present.

    This function is responsible for downloading the zip archive file.

    Parameters:
        url : str
            Remote URL of the dataset archive.
        download_path : Path
            Path to the downloaded dataset archive.

    Raises:
        RuntimeError
            If the download fails or the file cannot be written.
    """
    logger.info('Downloading Hollywood Heads Dataset')

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(download_path, 'wb') as out, tqdm(
            total=5.4*1024**3,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=download_path.name
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        bar.update(len(chunk))

    except Exception as e:
        logger.error(f'Error Occured while downloading the dataset: {str(e)}')
        raise RuntimeError('Download Failed') from e

def _extract_zip(zip_path: Path, raw_dir: Path):
    """
    Extract the dataset archive into the specified output directory.

    This function assumes the archive exists and is valid.

    Parameters:
        zip_path : Path
            Path to the dataset archive
        raw_dir : Path
            Directory where the archive will be extracted.

    Raises:
        RuntimeError:
            If file does not exist or extraction fails.
    """
    logger.info(f'Extracting the zip file at {zip_path} to {raw_dir}')

    try:
        with zipfile.ZipFile(zip_path) as z:
            members = z.infolist()
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                futures = [ex.submit(_extract, z, m, raw_dir) for m in members]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                    pass

        zip_path.unlink()

    except Exception as e:
        logger.error(f'Error occured during extraction: {str(e)}')
        raise RuntimeError('Extraction Failed') from e


def _create_split(data_dir: Path):
    """
    Create train/validation splits from the dataset.

    Parameters:
        data_dir: Path
            Root dataset directory.
    Raises:
        RuntimeError:
            If splitting fails.
    """
    logger.info(f'Creating train and validation dataset.')
    
    ANN_DIR = data_dir / 'HollywoodHeads' / 'Annotations'
    SPLIT_DIR = ANN_DIR.parent / 'Splits'

    try:
        if SPLIT_DIR.exists():
            logger.info('Split already exists')
            length = {}
            for split in ['train', 'val', 'test']:
                split_file = SPLIT_DIR / f'{split}.txt'
                length[split] = len(open(split_file, 'r').read().splitlines())
            logger.info(f"Dataset split into ==> train: {length['train']}, val: {length['val']}, test {length['test']}")

        else:
            
            if not SPLIT_DIR.is_dir():
                shutil.rmtree(SPLIT_DIR)

            SPLIT_DIR.mkdir(parents=True, exist_ok=True)

            # deterministic ordering
            filenames = sorted(
                f.stem for f in ANN_DIR.iterdir()
                if f.is_file() and f.suffix == ".xml"
            )

            # first split: train + temp
            train, val_test = train_test_split(
                filenames,
                test_size=0.04,
                random_state=42,
                shuffle=True,
            )

            val, test = train_test_split(
                val_test,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )

            (SPLIT_DIR / "train.txt").write_text("\n".join(train) + "\n")
            (SPLIT_DIR / "val.txt").write_text("\n".join(val) + "\n")
            (SPLIT_DIR / "test.txt").write_text("\n".join(test) + "\n")

            logger.info(f"Dataset split into ==> train: {len(train)}, val: {len(val)}, test {len(test)}")
    
    except Exception as e:
        logger.error(f'Error occured during splitting dataset: {str(e)}')
        raise RuntimeError('Splitting Failed') from e
    
    
def _if_dataset_exist(data_dir: Path) -> bool:
    
    base = data_dir / 'HollywoodHeads'
    
    if not base.exists():
        return False
        
    image_dir = base / 'JPEGImages'
    ann_dir = base / 'Annotations'
    
    if not image_dir.exists() or not ann_dir.exists():
        return False
    
    has_images = any(image_dir.iterdir())
    has_ann = any(ann_dir.iterdir())

    
    if not(has_images and has_ann):
        return False
    
    image_stems = {
        p.stem for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpeg"
    }

    ann_stems = {
        p.stem for p in ann_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".xml"
    }

    missing_annotations = image_stems - ann_stems
    missing_images = ann_stems - image_stems
    
    if missing_annotations:
        logger.warning(f'{len(missing_annotations)} .jpeg images without matching .xml')
        return False
    if missing_images:
        logger.warning(f'{len(missing_images)} .xml files without matching .jpeg')
        return False
        
    return True

def _clean_data_dir(data_dir: Path) -> None:
    for item in data_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

def collect_data(url: str, external_dir: Path, raw_data_dir: Path):
    """
    Orchestrates the full dataset collection stage.

    The function performs the following steps in order:
    1. Download the dataset if the archive file/ dataset is missing.
    2. Extract the dataset archive.
    3. Create train/val splits.

    If any steps fails, the execution is stopped and the error is logged.
    """
    logger.info('Starting Data Collection Stage')
    
    # url and data paths
    URL = url
    ZIP_PATH = external_dir / 'HollywoodHeads.zip'
    
    # If dataset already exists, skip everything
    if _if_dataset_exist(raw_data_dir):
        logger.info('Dataset already exists, skipping collection.')
        return
    
    try:
        logger.info('No valid dataset present, checking for archive file.')
        #_clean_data_dir(raw_data_dir)
        # download the dataset if 
        if not ZIP_PATH.exists():
            logger.info('No archive file found, attempting to download dataset.')
            if not url:
                raise ValueError('No url provided. Specify --url argument')
            _download_data(URL, ZIP_PATH)
        
        logger.info('Archive File Locally Saved')
        _extract_zip(ZIP_PATH, raw_data_dir)
        logger.info('Extraction Completed')
        logger.info("Checking dataset split.")
        time.sleep(10)
        _create_split(raw_data_dir)

    except Exception as e:
        logger.error(f'Runtime Error: {str(e)}')
        logger.info('Data Collection Stopped Abruptly!!, Check Log for Errors.')
        

if __name__ == '__main__':
    PROJ_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJ_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    collect_data(None, EXTERNAL_DATA_DIR, RAW_DATA_DIR)