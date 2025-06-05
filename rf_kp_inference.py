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
rf_model = joblib.load("rf_models/rf_kp_1.joblib")

video_path = "C:/wajahat/hand_in_pocket/test_bench/tp_t1.mp4"
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
    "position","kp_0_x", "kp_1_x", "kp_2_x", "kp_3_x", "kp_4_x", "kp_5_x", "kp_6_x", "kp_7_x", "kp_8_x", "kp_9_x",
    "kp_0_y", "kp_1_y", "kp_2_y", "kp_3_y", "kp_4_y", "kp_5_y", "kp_6_y", "kp_7_y", "kp_8_y", "kp_9_y"]

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

                feature_dict[f"kp_{i}_x"] = x
                feature_dict[f"kp_{i}_y"] = y

                # Draw keypoints
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # draw_lines(frame, keypoints, connections)
            distances = calculate_distances(keypoints, connections)
            feature_dict.update(distances)

            if len(keypoints) == 0 or all((x == 0 and y == 0) for x, y in keypoints):
                continue

            # Add position estimation (based on x of first keypoint)
            person_x = keypoints[0][0]
            if person_x < 460:
                position = 0
            elif 465 < person_x < 895:
                position = 1
            # elif 610 < person_x < 953:
            #     position = 0
            else:
                position = 2

            feature_dict['position'] = position

            # Create DataFrame and predict
            # input_df = pd.DataFrame([feature_dict])
            ordered_features = {key: feature_dict.get(key, 0.0) for key in feature_names}
            input_df = pd.DataFrame([ordered_features])
            # print(input_df)
            prediction = rf_model.predict(input_df)[0]

            # Draw prediction on the frame
            # label = "Hand in Pocket" if prediction == 1 else "No Hand in Pocket"
            # cv2.putText(frame, label, (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if prediction == 1:
                # label = "Hand in Pocket"
                cv2.putText(frame, "Hand in Pocket", (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            else:
                cv2.putText(frame, "No Hand in Pocket", (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
