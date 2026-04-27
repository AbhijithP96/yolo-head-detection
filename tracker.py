from ultralytics import YOLO
import pathlib
import cv2
import numpy as np
from collections import defaultdict, deque
from sort.sort import Sort

# the pytorch model is trained on windows, hence if you are running this on linux
# uncomment the below line
pathlib.WindowsPath = pathlib.PosixPath

_sort = Sort()
_model = YOLO("./models/yolov8n_head_detector.pt", verbose=False)

track_history = defaultdict(lambda: deque(maxlen=30))


def get_center(detection):

    x1, y1, x2, y2 = detection

    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def update_track_history(track_id, center):

    if track_history[track_id] and len(track_history[track_id]) > 0:
        track_history[track_id].append(center)

    else:
        track_history[track_id] = [center]


def track_with_sort():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output = _model(frame, verbose=False)

        detections = []
        pytorch_frame = frame.copy()

        for result in output:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                score = float(boxes.conf[i].cpu().numpy())
                detections.append([x1, y1, x2, y2, score])
                cv2.rectangle(
                    pytorch_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    pytorch_frame,
                    f"{score:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        sort_frame = frame.copy()

        dets_array = np.array(detections) if detections else np.empty((0, 5))
        tracker_bbs_ids = _sort.update(dets_array)

        for track in tracker_bbs_ids:
            x1, y1, x2, y2, track_id = track
            update_track_history(track_id, get_center((x1, y1, x2, y2)))
            cv2.rectangle(
                sort_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
            )
            cv2.putText(
                sort_frame,
                f"ID {int(track_id)}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        for track_id in track_history.keys():
            points = track_history[track_id]
            for i in range(1, len(points)):

                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))

                hue = int((track_id * 37) % 180)
                color = np.zeros((1, 1, 3), dtype=np.uint8)
                color[:, :, :] = [hue, 220, 255]
                color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR).tolist()[0][0]

                cv2.line(sort_frame, pt1, pt2, tuple(color), 1)

        cv2.imshow("PyTorch Detection", pytorch_frame)
        cv2.imshow("SORT Tracking", sort_frame),

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_with_sort()
