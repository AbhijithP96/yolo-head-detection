from pathlib import Path
import os
import yaml

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

# download url
try:
    URL = os.environ['URL']
except KeyError:
    # Load from params.yaml if env var not set
    params_path = PROJ_ROOT / "params.yaml"
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    URL = params.get('url', '')

logger.info(f'Download URL: {URL}')
# Paths
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
except ModuleNotFoundError:
    pass
