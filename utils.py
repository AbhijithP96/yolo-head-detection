import dagshub
import mlflow
import torch

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
        client = mlflow.MlflowClient()
        reg_model = client.get_registered_model("YoloHeadDetector")
        model_uri = f"models:/{reg_model.name}/production"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # unwrap the model to get the raw YOLOv8 model
        unwrapped_model = model.unwrap_python_model()
        yolo_model = unwrapped_model.get_raw_model()
        return yolo_model
    except Exception as e:
        raise RuntimeError("Failed to load the model from MLflow.") from e


def get_feat_maps(model, image):
    """
    Extract feature maps from the last 3 layers of the Yolo model for downstream tasks

    Args:
        model (_type_): Yolo Model
        image (_type_): Image whose features need to be extracted
    """

    features = {}

    def hook_fn(name):
        def hook_(module, input, output):
            output = output.clone().detach().cpu().numpy()
            features[name] = {"map": output.tolist(), "shape": list(output.shape)}

        return hook_

    backbone = model.model.model

    layers = [4, 6, 8]

    for layer in layers:
        backbone[layer].register_forward_hook(hook_fn(f"Layer:{layer}"))

    with torch.no_grad():
        _ = model(image)

    return features
