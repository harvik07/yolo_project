import os
dirname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dirname)

import cv2
from ultralytics import YOLO, settings
import torch
from time import perf_counter
import numpy as np

# 🔥 Disable unnecessary logging (faster)
settings.update({'mlflow': False})

# 🔥 Device setup (auto CPU/GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 🔥 Use lightweight model (FAST)
model = YOLO('yolo11n.pt').to(device)

# DeepSORT setup
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

deepsort = None

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=torch.cuda.is_available()  # ✅ FIXED
    )

init_tracker()

# 🔥 Video input
cap = cv2.VideoCapture('streams/1.mp4')

tracker_time = {}
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Resize (important for speed)
    frame = cv2.resize(frame, (640, 480))

    # 🔥 Skip alternate frames (2x speed)
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # 🔥 YOLO prediction (optimized)
    results = model.predict(frame, classes=[0], conf=0.4, verbose=False)

    for r in results:
        if len(r.boxes) > 0:
            bbox_xywh = r.boxes.xywh.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            outputs = deepsort.update(bbox_xywh, confs, clss, frame)

            if len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_id = map(int, output[:5])

                    # ⏱ Time tracking
                    if track_id not in tracker_time:
                        tracker_time[track_id] = [perf_counter(), perf_counter()]

                    tracker_time[track_id][1] = perf_counter()
                    time_in_store = tracker_time[track_id][1] - tracker_time[track_id][0]

                    # 🎯 Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 🆔 Show ID + Time
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {time_in_store:.1f}s",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

    # 🎥 Display
    cv2.imshow('Customer Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()