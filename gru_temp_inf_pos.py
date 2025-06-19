import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import json
from collections import deque
from ultralytics import YOLO
import pandas as pd

# ========================== GRU Model ==========================
class GRUClassifier(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(1)

# ========================== ROI Assignment ==========================
roi_data_list = []
def assign_roi_index(x):
    for roi in roi_data_list:
        if roi["xmin"] <= x < roi["xmax"]:
            return roi["position"]
    return None

# ========================== Load GRU Model ==========================
def load_gru_model(weights_path, device, input_size):
    model = GRUClassifier(input_size=input_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ========================== Matrices Rearranging ==========================
def rearrange_matrix(matrix):
    # assert matrix.shape == (5, 24), "Input must be of shape (5, 24)"
    # assert len(matrix) ==120

    original = np.array(matrix).reshape(5, 24)

    # Get keypoint indices for x and y
    x_indices = [i for i in range(0, 20, 2)]  # 0, 2, ..., 18
    y_indices = [i for i in range(1, 20, 2)]  # 1, 3, ..., 19
    position_indices = list(range(20, 24))   # 20, 21, 22, 23

    # New column order: all x's, then y's, then position data
    new_order = x_indices + y_indices + position_indices

    return original[:, new_order]

# ========================== Inference ==========================
if __name__ == "__main__":
    kp_model = YOLO("C:/wajahat/hand_in_pocket/bestv7-2.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WINDOW_SIZE = 5
    INPUT_SIZE = 24
    model_path = "rf_models/gru_temp_pos_gen.pth"
    gru_model = load_gru_model(model_path, device, INPUT_SIZE)

    # input_dir = "C:/wajahat/hand_in_pocket/test_bench"
    input_dir = "F:/Wajahat/hand_in_pocket/qiyas_test"
    json_path = "qiyas_multicam.camera_final.json"

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
            cam_id = input("Enter camera ID (or 's' to skip): ")
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
                    feature_vector = []

                    for i, keypoint in enumerate(kp_tensor[:10]):
                        x, y, conf = keypoint[:3].cpu().numpy()
                        if conf < 0.5:
                            x, y = -1, -1
                        else:
                            x /= 1280
                            y /= 720
                        feature_vector.extend([x, y])
                        keypoints.append((x, y))
                        if x != -1 and y != -1:
                            cv2.circle(frame, (int(x * 1280), int(y * 720)), 5, (0, 255, 0), -1)

                    if len(keypoints) == 0 or all((x == -1 and y == -1) for x, y in keypoints):
                        continue

                    person_x = keypoints[0][0] * 1280
                    position = assign_roi_index(person_x)
                    roi_data = roi_lookup.get(position)
                    if not roi_data:
                        continue

                    pos_list = roi_data.get("position_list", [0, 0, 0, 0])
                    position_feature = [float(p) for p in pos_list]
                    full_feature = feature_vector + position_feature

                    if position not in sliding_window:
                        sliding_window[position] = deque(maxlen=WINDOW_SIZE)
                    sliding_window[position].append(full_feature)

                    # print(f"Person {roi_data['desk']} at position {position} with features: {full_feature}")

                    if len(sliding_window[position]) == WINDOW_SIZE:
                        temporal_window = list(sliding_window[position])
                        # print(f"window: {temporal_window}")
                        input_array = rearrange_matrix(temporal_window)
                        # input_array = np.array(temporal_window)
                        # print(f"Input Array: {input_array}")
                        # input_array = input_array.reshape(WINDOW_SIZE, INPUT_SIZE)
                        input_seq = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)

                        # print(f"Input Sequence Shape: {input_seq.shape}")
                        # print(f"Input Sequence: {input_seq}")

                        with torch.no_grad():
                            prob = gru_model(input_seq).item()
                            prediction = 1 if prob >= 0.5 else 0

                        label = "Hand in Pocket" if prediction else "No Hand in Pocket"
                        color = (0, 0, 255) if prediction else (0, 255, 0)
                        cv2.putText(frame, f"{label} ({prob:.2f})", (int(person_x), 50 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Desk: {roi_data['desk']}, Pos: {roi_data['position']}",
                                    (int(person_x), 100 + person_idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 180, 0), 2)

                        cv2.imshow("GRU Inference", frame)
                        if prediction == 1:
                            cv2.waitKey(0)
                        else:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
            # cv2.imshow("GRU Inference", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
    cv2.destroyAllWindows()
