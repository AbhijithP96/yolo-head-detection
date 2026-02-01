import uvicorn
from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import numpy as np
from contextlib import asynccontextmanager

from utils import load_model

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["yolo_head_detector"] = load_model()
    yield
    models.clear()


app = FastAPI(
    title="YOLO Head Detection API",
    description="API for YOLO Head Detection Model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "YOLO Head Detection API is running."}


@app.post("/predict/file")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Predict heads in the uploaded image using the YOLO head detection model.

    Args:
        file (UploadFile): The image file uploaded by the user.
    Returns:
        dict: A dictionary containing prediction results.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((640, 640))
    image_np = np.array(image)

    model = models.get("yolo_head_detector")
    if model is None:
        return {"success": False, "error": "Model not loaded."}

    results = model(image_np)
    if not results or len(results) == 0:
        return {"success": False, "error": "No results from model."}

    predictions = {
        "boxes": results[0].boxes.xyxy.cpu().numpy().tolist(),
    }

    return {"success": True, "predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app)
