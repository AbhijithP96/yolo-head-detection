# YOLO Head Detection 🚀

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**YOLO-based human head detection** using the HollywoodHeads dataset and Ultralytics YOLOv8.
This repository contains data ingestion, preprocessing, training, experiment tracking (MLflow + DagsHub), and a DVC pipeline to reproduce runs and artifact management.

---

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Data & DVC Pipeline](#data--dvc-pipeline)
- [Experiments & Tracking (DagsHub / MLflow)](#experiments--tracking-dagshub--mlflow)
- [Running Training & Hyperparameter Search](#running-training--hyperparameter-search)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Project Structure](#project-structure)
- [Development & Contributing](#development--contributing)
- [License](#license)

---

## Features ✅

- End-to-end pipeline to collect and convert the HollywoodHeads dataset to YOLO format
- DVC pipeline (`dvc.yaml`) that defines data collection, validation, conversion, training, registration, and testing stages
- Experimentation utilities using MLflow and DagsHub for remote tracking and model registry
- Hyperparameter grid search scripts for reproducible experiments
- MLFlow model wrapper for easy model logging and inference

---

## Quickstart ⚡

Prerequisites:

- Python 3.11 (create a virtual environment)
- git
- dvc 
- Docker (optional, for reproducible CI builds)

Basic setup:

```bash
# Clone
git clone https://github.com/AbhijithP96/yolo-head-detection.git
cd yolo-head-detection

# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: install dvc
pip install dvc
```

Environment variables (recommended to set in CI secrets):

- `DAGSHUB_ACCESS_TOKEN` - token for DagsHub authentication. If not set, resorts to local MLflow tracking.
- `DAGSHUB_REPO_OWNER` - repo owner (defaults from `params.yaml`)
- `DAGSHUB_REPO_NAME` - repo name (defaults from `params.yaml`)
- `URL` - override dataset URL (defaults in `params.yaml`)

Check `params.yaml` for default training and repo values.

---
## Hugging Face Space

[Head Detection Demo](https://huggingface.co/spaces/abhiWanKenobi/yolo-head-detector)

You can find the gradio app code [here](https://github.com/AbhijithP96/detector-gradio)

---

## Data & DVC Pipeline 🔁

This project uses DVC to define and reproduce the data and training pipeline. The pipeline is defined in `dvc.yaml` with the following stages:

- `data_collection` — download HollywoodHeads (from `params.yaml` / `URL`) into `data/raw`
- `data_validation` — run basic validation and copy intermediate artifacts to `data/interim`
- `voc2yolo` — convert VOC-style annotations to YOLO format and save to `data/processed`
- `train` — run training (`yolo_head_detection/modeling/train.py`) and save `reports/run_info.json`
- `register` — register produced model artifacts and metadata into `reports/reg_details.json`
- `test` — run model evaluation producing `reports/test_results.json`

Typical DVC usage:

```bash
# Reproduce full pipeline locally (will run stages and generate outputs)
dvc repro

# If using a remote, fetch data & models
dvc pull

# After making changes, push data/artifacts to remote
dvc push
```

If you haven't configured a DVC remote yet, add one as usual:

```bash
dvc remote add -d myremote your-remote-url
```

---

## Experiments & Tracking — DagsHub (MLflow) 🔬

This repository logs experiments using MLflow and optionally pushes tracking data to DagsHub (if `DAGSHUB_*` env vars & token are set).

Key points:

- `yolo_head_detection/config.py` configures MLflow tracking URI. If `DAGSHUB_ACCESS_TOKEN` is available, the tracking URI is set to DagsHub's MLflow endpoint.
- `yolo_head_detection/modeling/train.py` logs training runs and the final model to MLflow.
- `yolo_head_detection/experiments/train_params.py` performs a hyperparameter grid search and logs nested MLflow runs for each trial.

Experiment example (hyperparameter search):

- Optimizers: `SGD`, `AdamW`
- Learning rates: `0.001`, `0.0001`
- Batch: `10`
- Epochs: `20`
- Fraction: `0.25` of the dataset (for quick experiments)

Run the hyperparameter search and push metrics to DagsHub/MLflow:

```bash
python yolo_head_detection/experiments/train_params.py
# Use --help to see available CLI options
```

Note: ensure `DAGSHUB_ACCESS_TOKEN`, `DAGSHUB_REPO_OWNER`, and `DAGSHUB_REPO_NAME` are set in environment or CI secrets to enable remote tracking.

Please find the experiments I conducted [here on DagsHub](https://dagshub.com/AbhijithP96/yolo-head-detection/experiments).

---

## Running Training & Evaluation ▶️

Train locally:

```bash
# To run the main training script via Typer-based CLI
python yolo_head_detection/modeling/train.py

# To run a test/evaluation after training
python yolo_head_detection/modeling/test.py
```

After successful training, run the registration step to capture model metadata:

```bash
python yolo_head_detection/modeling/register.py
```

Output artifacts and model weights are saved under `reports/` and `models/` as configured in `yolo_head_detection/config.py`.

---

## GitHub Actions CI/CD (example) 🛠️

Below is an example GitHub Actions workflow you can add at `.github/workflows/ci_cd.yml`. It checks for formatting, runs unit tests, and build and push the docker image to GitHub Container Registry.

You can change the workflow as needed and use the image name from `.github/workflows/ci_cd.yml` to deploy the API to cloud services.

Secrets to configure while deploying if using your own DagsHub repo:

- `DAGSHUB_ACCESS_TOKEN` - allows reading from Dagshub model registry.
- `DAGSHUB_REPO_OWNER` and `DAGSHUB_REPO_NAME` - repository identification.

---

## Project Structure (brief) 🗂️

```
<root>
├── data/              # raw/interim/processed datasets
├── models/            # included model weights (e.g. yolov8n.pt)
├── reports/           # training artifacts, figures, results
├── yolo_head_detection/
│   ├── dataset.py     # dataset ingestion, conversion, validation
│   ├── modeling/      # train/test/register/predict
│   └── experiments/   # hyperparameter search scripts
└── dvc.yaml           # DVC pipeline
```

---

## FastAPI Inference App 🚀

Run app.py to start a FastAPI server for inference:

```bash
# Start the API (loads model at startup)
python app.py
# or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
```

> Note: the server will attempt to load the registered model from MLflow/DagsHub using the `utils.load_model()` logic. 

---

## Dockerfile & Image (Inference) 🐳

A `dockerfile` is included for building a minimal FastAPI inference image. Build using the optional build args (useful to provide DagsHub credentials at build-time):

```bash
# Build with optional DagsHub args
docker build -f dockerfile -t yolo-head-detection-api

# Run the container and expose port 8000
docker run -p 8000:8000 \
  -e DAGSHUB_ACCESS_TOKEN=${DAGSHUB_ACCESS_TOKEN:-} \
  -e DAGSHUB_REPO_OWNER=${DAGSHUB_REPO_OWNER:-} \
  -e DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME:-} \
  yolo-head-detection-api
```

Troubleshooting tips:

- To use a local model artifact without DagsHub, modify `utils.load_model()` to load weights directly (e.g., from `models/` or `reports/`).

---

## API Endpoints 📡

- GET /
  - Health check. Returns a JSON `{ "message": "YOLO Head Detection API is running." }`.

- POST /predict/file
  - Description: Run head detection on an uploaded image.
  - Request: multipart/form-data with a field `file` containing the image (JPEG/PNG).
  - Response on success:
    ```json
    {
      "success": true,
      "predictions": {
        "boxes": [[x1, y1, x2, y2], ...]
      }
    }
    ```
  - Error response: `{ "success": false, "error": "message" }`

- POST /features/file
  - Description: Extract feature maps from internal YOLO layers for a given image.
  - Request: multipart/form-data with `file` image.
  - Response on success:
    ```json
    {
      "success": true,
      "features": {
        "Layer:4": {"shape": [N, C, H, W], "map": ...},
        "Layer:6": {"shape": ...},
        "Layer:8": {"shape": ...}
      }
    }
    ```

## `predict.py` Client Usage (CLI) 🧪

The repository provides a small Typer-based client script at `yolo_head_detection/modeling/predict.py` which demonstrates how to call the API.

- Send an image and display predictions:

```bash
python yolo_head_detection/modeling/predict.py main /path/to/image.jpg
```

This will:
- POST the image to `http://0.0.0.0:8000/predict/file` (the default `inference_url` in the script)
- Print predictions to stdout
- Draw boxes using OpenCV and show the annotated image in a window

- Extract feature map shapes:

```bash
python yolo_head_detection/modeling/predict.py features /path/to/image.jpg
```

- Real-time webcam inference (stream frames to the server):

```bash
python yolo_head_detection/modeling/predict.py camera
# Press 'q' to exit the webcam loop
```

Note: if your server runs on a different host/port, edit the `inference_url` variable inside `predict.py`.

---

## Final Notes & Troubleshooting 🔧

- If image previews do not appear when using `predict.py main`, ensure the environment has a GUI (or run inside an environment that supports OpenCV windows) or modify the script to save annotated outputs to disk.


