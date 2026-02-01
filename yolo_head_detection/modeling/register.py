"""
Model Registration Module

This module handles the registration of trained YOLO head detection models
to MLflow Model Registry on DagsHub. It retrieves model information from
the training run metadata and registers the model for versioning and deployment.
"""

from loguru import logger
from typer import Typer
import dagshub
import mlflow
import json

import sys
from pathlib import Path

# Add root directory to sys.path for importing project modules
root_path = Path(__file__).resolve().parents[2]  # Navigate up 2 levels to project root
sys.path.append(str(root_path))

from ultralytics import YOLO
from yolo_head_detection.config import TRACKING_URI, REPORTS_DIR

# Initialize Typer application for CLI
app = Typer()


@app.command()
def main():
    """
    Register the trained YOLO head detection model to MLflow Model Registry.

    This function performs the following steps:
    1. Establishes connection to MLflow tracking server on DagsHub
    2. Reads the training run information from the metadata JSON file
    3. Registers the trained model artifact with a versioned model name
    4. Transitions the registered model to 'staging' stage for testing

    Raises:
        RuntimeError: If any error occurs during the model registration process.
    """

    try:
        # Initialize DagsHub integration with MLflow for remote tracking
        logger.info("Setting up MLFlow Tracking on Dagshub")
        dagshub.init(
            repo_owner="AbhijithP96", repo_name="yolo-head-detection", mlflow=True
        )
        mlflow.set_tracking_uri(TRACKING_URI)

        # Load the training run metadata containing model URI and other run information
        run_info_json = REPORTS_DIR / "run_info.json"
        logger.info(f"Reading the training run info from {str(run_info_json)}")

        with open(run_info_json, "r") as f:
            run_info = json.load(f)

        # Register the model artifact from the MLflow run to the Model Registry
        run_id = run_info["run_id"]
        model_name = run_info["model_name"]
        model_uri = run_info["model_uri"]
        # model_uri = f'runs:/{run_id}/{model_name}'
        reg_model = mlflow.register_model(model_uri=model_uri, name="YoloHeadDetector")

        logger.info(
            f"Model {model_name} at {model_uri} registered as {reg_model.name} with version: {reg_model.version}."
        )

        # Transition the registered model to 'staging' stage for further validation and testing
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=reg_model.name,
            version=reg_model.version,
            stage="staging",
        )
        logger.info("Model has been moved to 'staging' stage for further testing.")

        reg_details = {
            "name": reg_model.name,
            "version": reg_model.version,
            "stage": reg_model.current_stage,
        }
        reg_details_json = REPORTS_DIR / "reg_details.json"
        with open(reg_details_json, "w") as reg:
            json.dump(reg_details, reg, indent=4)

        logger.info(f"Registration details saved at: {str(reg_details_json)}")

    except Exception as e:
        # Log error and raise RuntimeError for proper error handling
        logger.error("Error Occured during registering model")
        raise RuntimeError(str(e)) from e


if __name__ == "__main__":
    # Run the CLI application
    app()
