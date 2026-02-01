import dagshub
import mlflow

from yolo_head_detection.config import TRACKING_URI, repo_owner, repo_name


def load_model():
    """
    Load the YOLO head detection model from MLflow Model Registry.

    Returns:
        model: The loaded YOLO model.
    """
    try:
        # Initialize DagsHub and MLflow tracking
        if repo_owner and repo_name:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_tracking_uri(TRACKING_URI)

        # Load the registered model from production stage
        model_uri = "models:/YoloHeadDetector/production"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # unwrap the model to get the raw YOLOv8 model
        unwrapped_model = model.unwrap_python_model()
        yolo_model = unwrapped_model.get_raw_model()
        return yolo_model
    except Exception as e:
        raise RuntimeError("Failed to load the model from MLflow.") from e
