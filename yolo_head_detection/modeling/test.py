"""Model Evaluation Module

This module handles the evaluation of registered YOLO head detection models
on the test set. It retrieves the model from MLflow Model Registry (staging stage),
performs validation, and logs all evaluation metrics back to MLflow for tracking
and comparison.
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

from ultralytics import YOLO, settings
from yolo_head_detection.config import (
    TRACKING_URI,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TRACK
)

# Initialize Typer CLI application
app = Typer()


@app.command()
def main():
    """
    Evaluate the YOLO head detection model on the test set.

    This function performs the following workflow:
    1. Initializes MLflow tracking and connects to DagsHub
    2. Retrieves the registered model from staging stage
    3. Loads and unwraps the model to get the raw YOLOv8 model
    4. Runs validation on the test set with specified parameters
    5. Logs all evaluation metrics to MLflow for experiment tracking

    Raises:
        RuntimeError: If any error occurs during the evaluation process.
    """
    if not TRACK:
        raise NotImplementedError("Testing without MLFlow not implemented.")

    try:
        # Initialize DagsHub and MLflow tracking for remote experiment management
        logger.info("Setting up connection to model registry")
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name="eval_test_set")
        logger.info("MLflow experiment set to eval_test_set")

        # Create MLflow client for model registry operations
        client = mlflow.MlflowClient()

        # Fetch the registered model artifact from staging stage for evaluation
        logger.info("Retrieving registered model from staging stage")
        reg_model = client.get_registered_model(name="YoloHeadDetector")
        model_uri = f"models:/{reg_model.name}/staging"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # Unwrap the MLflow model wrapper to access the underlying YOLOv8 model
        unwarpped_model = model.unwrap_python_model()
        logger.info("Extracting raw YOLOv8 model from MLflow wrapper")
        yolo_model = unwarpped_model.get_raw_model()

        # update yolo settings to log in mlflow
        settings.update({"mlflow": False})

        # Evaluate the YOLO model on the test set and log metrics to MLflow
        with mlflow.start_run(run_name="testing") as run:
            # Run validation on test set with specified hyperparameters
            logger.info("Starting evaluation on test set")
            results = yolo_model.val(
                data=PROCESSED_DATA_DIR / "hollywoodheads.yaml",
                split="test",
                project=REPORTS_DIR,
                name="test",
                device=0,
                imgsz=640,
                batch=8,
                exist_ok=True,
            )

            # Extract and log all evaluation metrics to MLflow for tracking
            logger.info("Logging evaluation metrics to MLflow")

            # save the test_results
            test_result_json = REPORTS_DIR / "test_results.json"
            test_results = results.results_dict
            with open(test_result_json, "w") as res:
                json.dump(test_results, res, indent=4)

            mlflow.log_artifact(test_result_json)
            logger.info(
                f"Evaluation complete. Test Results Saved at {str(test_result_json)}"
            )

    except Exception as e:
        # Log error details and raise RuntimeError for proper error propagation
        logger.error(f"Error occurred during model evaluation: {str(e)}")
        raise RuntimeError(str(e)) from e


if __name__ == "__main__":
    # Execute the CLI application
    app()
