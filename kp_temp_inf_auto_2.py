# the change in this script is of the position values, it fetch the list of positions from the json file and then 
# parse it to the position_a, position_b, position_c, position_d values at the end of the feature vector

from ultralytics import YOLO
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from collections import deque
import json

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

roi_data_list = []

def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]

# Load models
if __name__ == "__main__":
    kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
    rf_model = joblib.load("rf_models/rf_temp_norm_pos_gen.joblib")  # your temporal model

    # input_dir = "C:/Users/LAMBDA THETA/Videos"
    input_dir = "C:/wajahat/hand_in_pocket/test_bench"
    json_path = "qiyas_multicam.camera_final.json"

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    if not video_files:
        print("No video files found in the directory.")
        exit()

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        connections = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 7), (4, 5), (5, 6), (7, 8), (8, 9)]

        single_frame_features = [f"kp_{i}_x" for i in range(10)] + [f"kp_{i}_y" for i in range(10)]
        WINDOW_SIZE = 5
        sliding_window = {}

        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Select Camera", frame)
        cv2.waitKey(1)

        with open(json_path, 'r') as f:
            camera_config = json.load(f)

        skip_video = False
        while True:
            camera_id_input = input("Enter camera ID: ")
            if camera_id_input.lower() == 's':
                print(f"Skipping video {video_path}")
                cap.release()
                cv2.destroyWindow("Select Camera")
                skip_video = True
                break
            else:
                camera_id = f"camera_{camera_id_input}"
                camera_data = next((cam for cam in camera_config if cam["_id"] == camera_id), None)
                if camera_data:
                    break
                print(f"Invalid camera ID {camera_id}. Please try again.")

        if skip_video:
            continue

        cv2.destroyWindow("Select Camera")
        roi_data_list = list(camera_data["data"].values())
        roi_lookup = {roi["position"]: roi for roi in roi_data_list}

        while True:
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
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        if conf < 0.5:
                            x, y = -1, -1
                        else:
                            x = x / 1280
                            y = y / 720

                        keypoints.append((x, y, conf))
                        feature_dict[f"kp_{i}_x"] = x
                        feature_dict[f"kp_{i}_y"] = y

                    if len(keypoints) == 0 or all((x == -1 and y == -1) for x, y, _ in keypoints):
                        continue

                    person_x = keypoints[0][0] * 1280
                    position = assign_roi_index(person_x)
                    roi_data = roi_lookup.get(position, None)
                    if roi_data is None:
                        print(f"ROI data not found for position {position}.")
                        continue

                    cv2.putText(frame, f"Desk: {roi_data['desk']}, Position: {roi_data['position']}",
                                (int(person_x), 100 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 180, 0), 2)

                    feature_dict['position'] = position
                    ordered_frame_features = {key: feature_dict.get(key, -1.0) for key in single_frame_features}

                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)

                    sliding_window[position].append(ordered_frame_features)

                    if len(sliding_window[position]) == WINDOW_SIZE:
                        flat_feature = {}
                        # Flatten temporal keypoints across t0–t4 (x and y only)
                        for t in range(WINDOW_SIZE):
                            for i in range(10):  # kp_0 to kp_9
                                x_key = f"kp_{i}_x"
                                flat_feature[f"{x_key}_t{t}"] = sliding_window[position][t].get(x_key, -1.0)
                            for i in range(10):  # kp_0 to kp_9
                                y_key = f"kp_{i}_y"
                                flat_feature[f"{y_key}_t{t}"] = sliding_window[position][t].get(y_key, -1.0)

                            # Add one-hot-like ROI position flags (a-d)
                        position_list = roi_data.get("position_list", [0, 0, 0, 0])
                        flat_feature["position_a"] = position_list[0]
                        flat_feature["position_b"] = position_list[1]
                        flat_feature["position_c"] = position_list[2]
                        flat_feature["position_d"] = position_list[3]

                        input_df = pd.DataFrame([flat_feature])
                        # print(f"Position: {position}, Desk: {roi_data['desk']}")
                        # print(input_df)

                        # input_df.to_csv("input_df.txt", index=True, sep='\t')

                        prediction = rf_model.predict(input_df)[0]

                        label = "Hand in Pocket" if prediction == 1 else "No Hand in Pocket"
                        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                        cv2.putText(frame, label, (int(person_x), 50 + person_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Temporal Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
