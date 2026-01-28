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

from yolo_head_detection.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR, TRACKING_URI

app = typer.Typer()

@app.command()
def main():
    
    cutmix = [1.0]
    
    logger.info('Setting MLFlow Tracking for Augmentation(CutMix) Search')
    dagshub.init(repo_owner='AbhijithP96', repo_name='yolo-head-detection', mlflow=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('cutmix')
    
    try:
    
        with mlflow.start_run(run_name='yolov8n-cutmix-search'):
            
            logger.info("Experiment Started")
            
            for ctmix in cutmix:
                
                run_name = f"run_cutmix:{ctmix}"
                logger.info(f"Starting run:{run_name}")
                with mlflow.start_run(nested=True, run_name=run_name):
                    
                    # Reinitialize model for each run to start fresh
                    model = YOLO(model=MODELS_DIR / 'yolov8n.pt')
                    logger.info("Model Yolov8n reinitialized for this run")
                    
                    params = {
                        'optimizer' : 'SGD',
                        'lr0' : 0.0001,
                        'batch' : 10,
                        'epochs' : 20,
                        'fraction' :0.25,
                        'patience' : 3,
                        'scale' : 0.7,
                        'mosaic' : 1.0,
                        'cutmix' : ctmix
                    }
                    
                    mlflow.log_params(params)
                    
                    results = model.train(
                        data=PROCESSED_DATA_DIR / 'hollywoodheads.yaml',
                        project=REPORTS_DIR,
                        name = run_name,
                        device = 0,
                        epochs = 20,
                        batch = 10,
                        lr0 = 0.0001,
                        optimizer = 'SGD',
                        fraction = 0.25,
                        patience=3,
                        scale = 0.7,
                        mosaic = 1.0,
                        cutmix = ctmix
                    )
                    
                    mlflow.log_metrics({
                                        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                                        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                                        'precision': results.results_dict.get('metrics/precision(B)', 0),
                                        'recall': results.results_dict.get('metrics/recall(B)', 0),
                                        'fitness': results.results_dict.get('fitness', 0)
                                    })
        logger.info('Experiment Finished Successfully')
    except Exception as e:
        logger.error('Error Occured During Experimentation')
        raise RuntimeError(str(e)) from e
        
    
if __name__ == '__main__':
    app()