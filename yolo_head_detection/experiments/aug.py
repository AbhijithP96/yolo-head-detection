import mlflow
import dagshub
import typer

from loguru import logger
from ultralytics import YOLO

import sys
from pathlib import Path

# Add root/ to sys.path
root_path = Path(__file__).resolve().parents[2]  # go up 2 levels from train.py
sys.path.append(str(root_path))

from yolo_head_detection.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    TRACKING_URI,
)

app = typer.Typer()


@app.command()
def main():
    """
    Perform hyperparameter tuning for data augmentation parameters in YOLOv8n model training.

    This function conducts a grid search over specified mosaic and scale augmentation values,
    training the YOLOv8n model for each combination. It uses MLflow for experiment tracking
    and Dagshub for remote logging. Each augmentation combination is run as a nested MLflow
    run under a parent experiment.

    Hyperparameters tuned:
    - Mosaic: 1.0
    - Scale: 0.5, 0.7, 0.9

    Fixed parameters:
    - Optimizer: SGD
    - Learning rate: 0.0001
    - Batch size: 10
    - Epochs: 20
    - Fraction: 0.25 (fraction of dataset to use)
    - Patience: 3 (early stopping patience)

    Metrics logged:
    - mAP50
    - mAP50-95
    - Precision
    - Recall
    - Fitness

    Raises:
        RuntimeError: If an error occurs during experimentation, wraps the original exception.
    """
    mosaics = [1.0]
    scales = [0.5, 0.7, 0.9]

    logger.info("Setting MLFlow Tracking for Augmentation Search")
    dagshub.init(repo_owner="AbhijithP96", repo_name="yolo-head-detection", mlflow=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("augmentations")

    try:
        with mlflow.start_run(run_name="yolov8n-aug-search"):
            logger.info("Experiment Started")

            for mosaic in mosaics:
                for scale in scales:
                    run_name = f"run_mosaic:{mosaic}_scale:{scale}"
                    logger.info(f"Starting run:{run_name}")
                    with mlflow.start_run(nested=True, run_name=run_name):
                        # Reinitialize model for each run to start fresh
                        model = YOLO(model=MODELS_DIR / "yolov8n.pt")
                        logger.info("Model Yolov8n reinitialized for this run")

                        params = {
                            "optimizer": "SGD",
                            "lr0": 0.0001,
                            "batch": 10,
                            "epochs": 20,
                            "fraction": 0.25,
                            "patience": 3,
                            "scale": scale,
                            "mosaic": mosaic,
                        }

                        mlflow.log_params(params)

                        results = model.train(
                            data=PROCESSED_DATA_DIR / "hollywoodheads.yaml",
                            project=REPORTS_DIR,
                            name=run_name,
                            device=0,
                            epochs=20,
                            batch=10,
                            lr0=0.0001,
                            optimizer="SGD",
                            fraction=0.25,
                            patience=3,
                            scale=scale,
                            mosaic=mosaic,
                        )

                        mlflow.log_metrics(
                            {
                                "mAP50": results.results_dict.get(
                                    "metrics/mAP50(B)", 0
                                ),
                                "mAP50-95": results.results_dict.get(
                                    "metrics/mAP50-95(B)", 0
                                ),
                                "precision": results.results_dict.get(
                                    "metrics/precision(B)", 0
                                ),
                                "recall": results.results_dict.get(
                                    "metrics/recall(B)", 0
                                ),
                                "fitness": results.results_dict.get("fitness", 0),
                            }
                        )
        logger.info("Experiment Finished Successfully")
    except Exception as e:
        logger.error("Error Occured During Experimentation")
        raise RuntimeError(str(e)) from e


if __name__ == "__main__":
    app()
