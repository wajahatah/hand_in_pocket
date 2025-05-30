from ultralytics import YOLO
import os
import cv2
import numpy as np
import joblib
import pandas as pd

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load models
kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
rf_model = joblib.load("rf_models/rf_ds_1.joblib")

video_path = "F:/Wajahat/hand_in_pocket/hand_in_pocket/cam_1/chunk_26-02-25_10-15-desk1-2-3.avi"
cap = cv2.VideoCapture(video_path)

def draw_lines(frame, keypoints, connections):
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

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

# Keypoint connections for drawing and distance calculation
connections = [
    (0, 1), (0, 2), (0, 3), 
    (1, 4), (1, 7), 
    (4, 5), (5, 6), 
    (7, 8), (8, 9)
]

feature_names = [
    "position","distance(0,1)", "distance(0,2)", "distance(0,3)", "distance(1,4)", "distance(1,7)",
    "distance(4,5)", "distance(5,6)", "distance(7,8)", "distance(8,9)"]

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
                    x, y = 0, 0  # Optionally zero out low-confidence points
                keypoints.append((x, y))

                # feature_dict[f"kp_{i}_x"] = x
                # feature_dict[f"kp_{i}_y"] = y

                # Draw keypoints
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # draw_lines(frame, keypoints, connections)
            distances = calculate_distances(keypoints, connections)
            feature_dict.update(distances)

            if len(keypoints) == 0 or all((x == 0 and y == 0) for x, y in keypoints):
                continue

            # Add position estimation (based on x of first keypoint)
            person_x = keypoints[0][0]
            if person_x < 363:
                position = -1
            elif 365 < person_x < 728:
                position = 0
            elif 730 < person_x: 
            # < 973:
                position = 1
            # else:
            #     position = 2

            feature_dict['position'] = position

            # Create DataFrame and predict
            # input_df = pd.DataFrame([feature_dict])
            ordered_features = {key: feature_dict.get(key, 0.0) for key in feature_names}
            input_df = pd.DataFrame([ordered_features])
            prediction = rf_model.predict(input_df)[0]

            # Draw prediction on the frame
            label = "Hand in Pocket" if prediction == 1 else "No Hand in Pocket"
            cv2.putText(frame, label, (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
