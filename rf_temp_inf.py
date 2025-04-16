from ultralytics import YOLO
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from collections import deque

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load models
kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
rf_model = joblib.load("rf_models/rf_3.joblib")  # your temporal model

video_path = "C:/wajahat/hand_in_pocket/test_bench/cam_6_t1.mp4"
#"F:/Wajahat/hand_in_pocket/hand_in_pocket/cam_1/chunk_26-02-25_10-15-desk1-2-3.avi"
cap = cv2.VideoCapture(video_path)

# Connections and feature names
connections = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 7), (4, 5), (5, 6), (7, 8), (8, 9)]

single_frame_features = [
    f"kp_{i}_x" for i in range(10)
] + [
    f"kp_{i}_y" for i in range(10)
] + [
    "distance(0,1)", "distance(0,2)", "distance(0,3)", "distance(1,4)", "distance(1,7)",
    "distance(4,5)", "distance(5,6)", "distance(7,8)", "distance(8,9)", "position"
]

# Define temporal window size
WINDOW_SIZE = 5
sliding_window = deque(maxlen=WINDOW_SIZE)

def calculate_distances(keypoints, connections):
    distances = {}
    for (i, j) in connections:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[j]
            dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            distances[f"distance({i},{j})"] = dist
        else:
            distances[f"distance({i},{j})"] = 0.0
    return distances

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    results = kp_model(frame)

    for result in results:
        keypoints_tensor = result.keypoints.data
        for person_idx, kp_tensor in enumerate(keypoints_tensor):
            keypoints = []
            feature_dict = {}

            for i, keypoint in enumerate(kp_tensor):
                x, y, conf = keypoint[:3].cpu().numpy()
                if conf < 0.5:
                    x, y = 0, 0
                keypoints.append((x, y))
                feature_dict[f"kp_{i}_x"] = x
                feature_dict[f"kp_{i}_y"] = y
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            distances = calculate_distances(keypoints, connections)
            feature_dict.update(distances)

            if len(keypoints) == 0 or all((x == 0 and y == 0) for x, y in keypoints):
                continue

            person_x = keypoints[0][0]
            if person_x < 315:
                position = -2
            elif 320 < person_x < 667:
                position = -1
            elif 670 < person_x < 1015:
                position = 0
            else:
                position = 2
            feature_dict['position'] = position

            ordered_frame_features = {key: feature_dict.get(key, 0.0) for key in single_frame_features}
            sliding_window.append(ordered_frame_features)

            # Predict only if window is full
            if len(sliding_window) == WINDOW_SIZE:
                flat_feature = {}
                for idx, frame_feats in enumerate(sliding_window):
                    for key, value in frame_feats.items():
                        flat_feature[f"{key}_t{idx}"] = value
                input_df = pd.DataFrame([flat_feature])
                print(input_df)
                prediction = rf_model.predict(input_df)[0]

                label = "Hand in Pocket" if prediction == 1 else "No Hand in Pocket"
                cv2.putText(frame, label, (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Temporal Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
