import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from collections import deque
from ultralytics import YOLO

# ========== MLP Model ==========
class MLP(nn.Module):
    def __init__(self, input_size=105, hidden_size=64):
        super(MLP, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size // 2, 1),
        #     nn.Sigmoid()
        # )

        # for c2 model architecture
        self.net = nn.Sequential(
            nn.Linear(105, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ========== ROI Assignment ==========
roi_data_list = []
def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]

# ========== Load Model ==========
def load_mlp_model(weights_path, device):
    model = MLP()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ========== Main Inference ==========
if __name__ == "__main__":
    kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_mlp_model("rf_models/mlp_temp_norm_regrouped_l1_v2-c2.pt", device)

    # input_dir = "C:/Users/LAMBDA THETA/Videos"
    input_dir = "C:/wajahat/hand_in_pocket/test_bench"
    json_path = "qiyas_multicam.camera_final.json"
    WINDOW_SIZE = 5

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    if not video_files:
        print("No videos found.")
        exit()

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error loading video.")
            continue

        # Get camera ID
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Select Camera", frame)
        cv2.waitKey(1)

        with open(json_path, 'r') as f:
            camera_config = json.load(f)

        skip_video = False
        while True:
            cam_id = input("Enter camera ID: ")
            if cam_id.lower() == 's':
                skip_video = True
                cap.release()
                cv2.destroyWindow("Select Camera")
                break
            cam_key = f"camera_{cam_id}"
            camera_data = next((cam for cam in camera_config if cam["_id"] == cam_key), None)
            if camera_data:
                break
            print("Invalid camera ID. Try again.")

        if skip_video:
            continue

        cv2.destroyWindow("Select Camera")
        roi_data_list = list(camera_data["data"].values())
        roi_lookup = {roi["position"]: roi for roi in roi_data_list}
        sliding_window = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = kp_model(frame)

            for result in results:
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue

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
                        feature_dict[f"kp_{i}_x"] = x
                        feature_dict[f"kp_{i}_y"] = y
                        keypoints.append((x, y))

                    if len(keypoints) == 0 or all((x == -1 and y == -1) for x, y in keypoints):
                        continue

                    person_x = keypoints[0][0] * 1280
                    position = assign_roi_index(person_x)
                    roi_data = roi_lookup.get(position)
                    if not roi_data:
                        continue

                    feature_dict['position'] = position
                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)
                    sliding_window[position].append(feature_dict)

                    if len(sliding_window[position]) == WINDOW_SIZE:
                        flat_feature = {}
                        for i in range(10):
                            for axis in ['x', 'y']:
                                for t in range(WINDOW_SIZE):
                                    flat_feature[f"kp_{i}_{axis}_t{t}"] = sliding_window[position][t][f"kp_{i}_{axis}"]
                        for t in range(WINDOW_SIZE):
                            flat_feature[f"position_t{t}"] = sliding_window[position][t]["position"]

                        ordered_columns = [f"kp_{i}_{axis}_t{t}" for i in range(10) for axis in ['x', 'y'] for t in range(WINDOW_SIZE)]
                        ordered_columns += [f"position_t{t}" for t in range(WINDOW_SIZE)]
                        input_tensor = torch.tensor([[flat_feature[col] for col in ordered_columns]], dtype=torch.float32).to(device)

                        # print(f"Position: {position}")
                        # print(input_tensor)

                        with torch.no_grad():
                            prob = mlp_model(input_tensor).item()
                            prediction = 1 if prob >= 0.5 else 0

                        label = "Hand in Pocket" if prediction else "No Hand in Pocket"
                        color = (0, 0, 255) if prediction else (0, 255, 0)
                        cv2.putText(frame, f"{label} ({prob:.2f})", (int(person_x), 50 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Desk: {roi_data['desk']}, Pos: {roi_data['position']}",
                                    (int(person_x), 100 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 180, 0), 2)

            cv2.imshow("MLP Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    cv2.destroyAllWindows()
