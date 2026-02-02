"""Promote a staged MLflow model to production.

This module provides a small CLI to promote the latest model version in the
`staging` stage of the MLflow Model Registry to the `production` stage. It
connects to MLflow (optionally via DagsHub integration) and transitions the
model version, archiving any existing production versions.

Usage:
    python yolo_head_detection/modeling/prod.py main
"""

from loguru import logger
from typer import Typer
import dagshub
import mlflow

import sys
from pathlib import Path

# Add root directory to sys.path for importing project modules
root_path = Path(__file__).resolve().parents[2]  # Navigate up 2 levels to project root
sys.path.append(str(root_path))

from ultralytics import YOLO
from yolo_head_detection.config import TRACKING_URI, REPORTS_DIR, TRACK

app = Typer()


@app.command()
def main():
    
    if not TRACK:
        logger.warning(f"Using model registry at {TRACKING_URI}")
    try:
        # Initialize DagsHub integration with MLflow for remote tracking
        logger.info("Setting up MLFlow Tracking on Dagshub")
        mlflow.set_tracking_uri(TRACKING_URI)

        client = mlflow.MlflowClient()

        # Retrieve the registered model and its latest staged version
        logger.info("Retrieving registered model currently in 'staging' stage")
        reg_model = client.get_registered_model(name="YoloHeadDetector")

        staged_versions = client.get_latest_versions(
            name=reg_model.name, stages=["staging"]
        )
        if not staged_versions:
            raise RuntimeError(f"No staged versions found for model {reg_model.name}")

        reg_model_version = staged_versions[0]

        logger.info(
            f"Transitioning model {reg_model.name} version {reg_model_version.version} to 'production' stage."
        )

        # Promote the staged version to production and archive existing production versions
        client.transition_model_version_stage(
            name=reg_model.name,
            version=reg_model_version.version,
            stage="production",
            archive_existing_versions=True,
        )

        logger.info("Model promotion to production completed.")

    except Exception as e:
        logger.error(f"Error during model promotion to production: {e}")
        raise RuntimeError("Model promotion to production failed.") from e


if __name__ == "__main__":
    # Run the CLI application
    app()
