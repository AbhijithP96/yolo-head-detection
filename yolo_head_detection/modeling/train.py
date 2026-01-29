from loguru import logger
from tqdm import tqdm
import dagshub
import mlflow
import json
import typer
import numpy as np
from PIL import Image

import sys
from pathlib import Path

# Add root/ to sys.path
root_path = Path(__file__).resolve().parents[2]  # go up 2 levels from train.py
sys.path.append(str(root_path))

from ultralytics import YOLO, settings
from yolo_head_detection.config import MODELS_DIR, PROCESSED_DATA_DIR, TRAINER, TRACKING_URI, REPORTS_DIR

app = typer.Typer()

class YoloMlFlowModel(mlflow.pyfunc.PythonModel):
    """MLFlow wrapper for YOLO object detection model.
    
    This class provides a custom MLFlow PythonModel that wraps a YOLO model,
    enabling it to be logged, versioned, and deployed through MLFlow.
    """
    
    def load_context(self, context):
        """Load the YOLO model from artifacts.
        
        Args:
            context: MLFlow context containing model artifacts.
        """
        self.model = YOLO(context.artifacts["yolo_model"])

    def predict(self, context, model_input):
        """Run predictions on input images.
        
        Args:
            context: MLFlow context (unused).
            model_input: Batch of input images as numpy arrays.
            
        Returns:
            List of bounding box predictions in xywh format for each image.
        """
        model_input = [img for img in model_input]
        results = self.model(model_input)
        # Collect boxes for all results
        outputs = [r.boxes.xywh.cpu().numpy() for r in results if r.boxes is not None]
        return outputs
    
    def get_raw_model(self):
        """Get the underlying YOLO model.
        
        Returns:
            The YOLO model instance.
        """
        return self.model

@app.command()
def main():
    """Train a YOLO model for head detection and log results to MLFlow.
    
    This function orchestrates the complete training pipeline:
    1. Initializes MLFlow tracking on DagsHub
    2. Loads and trains a YOLO model
    3. Logs the best model to MLFlow with signature and example inputs
    4. Saves run information to a JSON file
    
    Raises:
        RuntimeError: If any error occurs during training.
    """
    try:
    
        logger.info('Setting up MLFlow Tracking on Dagshub')
        dagshub.init(repo_owner='AbhijithP96', repo_name='yolo-head-detection', mlflow=True)
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(TRAINER.exp)
        
        # Disable YOLO's automatic MLFlow logging to control manually
        settings.update({'mlflow': False})
        
        with mlflow.start_run(run_name=TRAINER.run) as run:
            # Log training parameters
            mlflow.log_params(vars(TRAINER))
            
            logger.info('Initializing Yolo Model')
            model = YOLO(
                model=MODELS_DIR / TRAINER.model,
                verbose=True
            )
            
            logger.info('Training Started...')
            results = model.train(
                data=PROCESSED_DATA_DIR / 'hollywoodheads.yaml',
                epochs=TRAINER.epochs,
                batch=TRAINER.batch,
                project=REPORTS_DIR,
                name=TRAINER.run,
                device=0,
                patience=TRAINER.patience,
                scale=TRAINER.scale,
                mosaic=TRAINER.mosaic,
                cutmix=TRAINER.cutmix,
                optimizer=TRAINER.optimizer,
                lr0=TRAINER.lr0,
                fraction=1.0
            )
            
            
            logger.info('Logging the model to mlflow')
            weights_path = REPORTS_DIR / TRAINER.run / 'weights' / 'best.pt'
            final_model = YOLO(weights_path)
            model_input = np.array(Image.open(REPORTS_DIR / 'figures' / 'mov_019_022253.jpeg').convert('RGB').resize((640,640)))
            
            batch_input = np.expand_dims(model_input, axis=0)  # shape: (1, H, W, C)
            batch_output = [r.boxes.xywh.cpu().numpy() for r in final_model([img for img in batch_input]) if r.boxes is not None]
            
            model_signature = mlflow.models.infer_signature(
                model_input=batch_input,
                model_output=batch_output)
            
            model_info = mlflow.pyfunc.log_model(
                name='yolo_head_detection',
                python_model=YoloMlFlowModel(),
                artifacts={"yolo_model": str(weights_path)},
                signature=model_signature,
                input_example=batch_input
            )
            
        logger.info('Training Completed.')
        # save the run info to a json file
        run_info = {
            'run_id' : run.info.run_id,
            'model_name' : 'yolo_head_detection',
            'model_uri' : model_info.model_uri,
        }
        logger.info('Saving run info to reports/run_info.json')
        reports_path = REPORTS_DIR / 'run_info.json'
        with open(reports_path, 'w') as f:
            json.dump(run_info, f, indent=4)
        
    except Exception as e:
        logger.error('Error Occured during training')
        raise RuntimeError(str(e)) from e

if __name__ == "__main__":
    app()
