from pathlib import Path
import typer

import requests
import base64
from io import BytesIO
from PIL import Image

import cv2
import numpy as np

app = typer.Typer()
inference_url = "http://127.0.0.1:8000/predict/file"


@app.command()
def main(image_path: str):

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    files = {"file": (Path(image_path).name, base64.b64decode(img_str), "image/jpeg")}
    response = requests.post(inference_url, files=files)

    if response.status_code == 200:
        output = response.json()
        if output["success"]:
            print("Predictions:")
            print(output["predictions"])
            boxes = output["predictions"]["boxes"]
            image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            cv2.imshow("box", image)
            cv2.waitKey(0)

        else:
            print("Error in prediction:", output["error"])
    else:
        print(f"Request failed with status code {response.status_code}")


@app.command()
def camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        _, img_encoded = cv2.imencode(".jpeg", frame)
        files = {"file": ("frame.jpeg", img_encoded.tobytes(), "image/jpeg")}
        response = requests.post(inference_url, files=files)

        if response.status_code == 200:
            output = response.json()
            if output["success"]:
                boxes = output["predictions"]["boxes"]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
