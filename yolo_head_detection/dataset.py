from pathlib import Path

from loguru import logger
import typer

from config import INTERIM_DATA_DIR,PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, URL
from data_utils import collect_data, validate_data, convert_voc_to_yolo

app = typer.Typer()


@app.command()
def collect(url: str = URL):
    """
    Collect, extract raw data from the specified URL.
    """
    if not url:
        logger.warning('No url provided for download. Checking for dataset on local machine.')
    
    collect_data(url, EXTERNAL_DATA_DIR, RAW_DATA_DIR)


@app.command()
def validate():
    """
    Validate the collected data.
    """
    validate_data(RAW_DATA_DIR, INTERIM_DATA_DIR)
    
@app.command()
def convert():
    """
    Convert VOC format to YOLO format
    """
    
    convert_voc_to_yolo(RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR)


@app.command()
def main(url: str = URL):
    """
    Run the full data pipeline: collection and validation.
    """
    collect(url)
    validate()


if __name__ == "__main__":
    app()
