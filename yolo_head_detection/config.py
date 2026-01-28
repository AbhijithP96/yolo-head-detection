"""Configuration module for YOLO head detection project.

This module initializes project paths, loads configuration parameters from params.yaml,
and sets up environment variables for the training pipeline.
"""

from pathlib import Path
import os
import yaml

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Project root directory (parent of the yolo_head_detection package)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Load parameters from params.yaml configuration file
params_path = PROJ_ROOT / "params.yaml"
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

# Load download URL and MLFlow tracking URI from environment variables or params.yaml
try:
    URL = os.environ['URL']
    TRACKING_URI = os.environ['TRACKING_URI']
except KeyError:
    # Fallback to params.yaml if environment variables are not set
    URL = params.get('url', '')
    TRACKING_URI = params.get('tracking_uri', 'http://localhost:5000')

logger.info(f'Download URL: {URL}')

# Define project directory structure
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Directory for storing trained models
MODELS_DIR = PROJ_ROOT / "models"

# Directories for storing reports and training results
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru to output through tqdm.write
# This ensures log messages don't interfere with tqdm progress bars
# Reference: https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
except ModuleNotFoundError:
    pass


class Trainer:
    """Configuration class for training parameters.
    
    This class encapsulates all hyperparameters and settings needed for training
    the YOLO model, including optimizer settings, data augmentation parameters,
    and learning rate configuration.
    
    Attributes:
        exp (str): Experiment name for tracking.
        run (str): Run name for this specific training run.
        model (str): Path to the model checkpoint to use.
        optimizer (str): Optimizer type (SGD, Adam, etc.).
        batch (int or float): Batch size for training.
        epochs (int): Number of training epochs.
        scale (float): Scale factor for data augmentation.
        mosaic (float): Mosaic augmentation probability (0.0 to 1.0).
        lr0 (float): Initial learning rate.
        cutmix (float): Cutmix augmentation probability (0.0 to 1.0).
    """
    
    def __init__(self, training_param_dict):
        """Initialize trainer with parameters from a dictionary.
        
        Args:
            training_param_dict (dict): Dictionary containing training parameters
                loaded from params.yaml.
        """
        self.exp = training_param_dict.get('exp', 'Default-exp')
        self.run = training_param_dict.get('run', 'default-run')
        self.model = training_param_dict.get('model', 'yolov8n.pt')
        self.optimizer = training_param_dict.get('optimizer', 'SGD')
        
        # Handle batch size - if value >= 1, convert to int; otherwise treat as fraction
        batch = float(training_param_dict.get('batch', 16))
        self.batch = batch if batch < 1 else int(batch)
        
        self.epochs = int(training_param_dict.get('epochs', 30))
        self.patience = int(training_param_dict.get('patience', 5))
        
        # Data augmentation parameters
        self.scale = float(training_param_dict.get('scale', 0.8))
        self.mosaic = float(training_param_dict.get('mosaic', 1.0))
        self.lr0 = float(training_param_dict.get('lr0', 0.01))
        self.cutmix = float(training_param_dict.get('cutmix', 0.0))
        
# Initialize trainer with parameters from params.yaml
TRAINER = Trainer(params['training'])