# ========= INFERENCE SCRIPT FOR 3D CNN FUSION MODEL =========

import os
import cv2
import json
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
# from model_hybrid_tempkpcrop import Fusion3DCNNKeypointModel  # <-- Use your 3D model class

# ========= CONFIG =========
YOLO_MODEL_PATH = "bestv7-2.pt"
FUSION_MODEL_PATH = "rf_models/hybrid_tempkpcrop_model3-1.pth"
JSON_PATH = "qiyas_multicam.camera_final.json"
VIDEO_DIR = "C:/wajahat/hand_in_pocket/test_bench"
CROP_SIZE = (64, 64)
WINDOW_SIZE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= LOAD MODELS =========
print("Loading YOLO keypoint model...")
kp_model = YOLO(YOLO_MODEL_PATH).to(DEVICE)

# ========== 3D CNN Fusion Model Architecture =========
import torch
import torch.nn as nn

# ==== 3D CNN Encoder ====
class Crop3DCNN(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),  # (B, 1, 5, 64, 64) -> (B, 8, 5, 64, 64)
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),                   # (B, 8, 5, 32, 32)
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),                   # (B, 16, 5, 16, 16)
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: (B, T, 1, H, W) → expected by 3D CNN as (B, C, D, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # → (B, 1, 5, 64, 64)
        x = self.features(x)
        x = self.fc(x)
        return x  # (B, out_dim)


# ==== Keypoint MLP Encoder ====
class KeypointMLP(nn.Module):
    def __init__(self, in_dim=101, out_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):  # x: (B, 101)
        return self.model(x)


# ==== Fusion Model ====
class Fusion3DCNNKeypointModel(nn.Module):
    def __init__(self, crop_feat_dim=32, kpt_feat_dim=32):
        super().__init__()
        self.cnn = Crop3DCNN(out_dim=crop_feat_dim)
        self.kpt = KeypointMLP(in_dim=101, out_dim=kpt_feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(crop_feat_dim + kpt_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # binary classification
        )

    def forward(self, crops, keypoints):
        crop_feat = self.cnn(crops)        # (B, crop_feat_dim)
        kpt_feat = self.kpt(keypoints)     # (B, kpt_feat_dim)
        fused = torch.cat([crop_feat, kpt_feat], dim=-1)
        return self.classifier(fused)      # (B, 2)


# ========= Load Fusion Model ========= 
print("Loading 3D CNN Fusion model...")
fusion_model = Fusion3DCNNKeypointModel().to(DEVICE)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
fusion_model.eval()

# ========= UTILS =========
def get_position(x, roi_data_list):
    for roi in roi_data_list:
        if roi['xmin'] <= x <= roi['xmax']:
            return roi['position']
    return None

# ========= INFERENCE =========
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"\n🎥 Processing: {video_file}")
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read video.")
        continue
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Select Camera", frame)
    cv2.waitKey(1)

    with open(JSON_PATH, "r") as f:
        camera_config = json.load(f)

    skip_video = False
    while True:
        cam_id = input("Enter camera ID for this video (or 's' to skip): ")
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

    roi_data_list = list(camera_data['data'].values())
    roi_lookup = {roi['position']: roi for roi in roi_data_list}
    desk_data = camera_data['data']
    sliding_window = {roi['position']: deque(maxlen=WINDOW_SIZE) for roi in roi_data_list}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        display_frame = frame.copy()

        results = kp_model(frame)
        for result in results:
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                continue
            keypoint_tensor = result.keypoints.data

            for kp_tensor in keypoint_tensor:
                keypoints = []
                keypoints_c = []
                for kp in kp_tensor[:10]:
                    x, y, conf = kp[:3].cpu().numpy()
                    if conf < 0.5:
                        x, y = 0, 0
                    keypoints_c.append((x, y))
                    keypoints.extend([x, y])

                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.circle(display_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                if not keypoints_c or all(x == 0 and y == 0 for x, y in keypoints_c):
                    continue

                person_x = keypoints_c[0][0]
                position = get_position(person_x, roi_data_list)
                if position is None:
                    continue

                print(f"Detected position: {position}")

                roi = roi_lookup.get(position)
                if not roi:
                    continue
                
                # for desk_id, roi in desk_data.items():
                # for roi in roi_data_list:
                #     xmin, xmax = roi['xmin'], roi['xmax']
                #     ymin, ymax = roi['ymin'], roi['ymax']
                #     desk_num = roi['desk']

                xmin, xmax = roi['xmin'], roi['xmax']
                ymin, ymax = roi['ymin'], roi['ymax']
                desk_num = roi['desk']

                print(f"crop desk: {desk_num}")
                crop = frame[:, xmin:xmax]  # Fixed cropping region
                cv2.imshow(f"desk_{desk_num}", crop)

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, CROP_SIZE)
                norm_crop = gray_resized.astype(np.float32) / 255.0
                crop_tensor = torch.tensor(norm_crop, device=DEVICE).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)


                keypoint_tensor = torch.tensor(keypoints, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                # position_tensor = torch.tensor([roi['position']], dtype=torch.float32, device=DEVICE)
                # full_kpt = torch.cat([keypoint_tensor.view(-1), position_tensor]).unsqueeze(0)  # (1, 101)

                # print(f"position: {position}/n, Full_kp: {full_kpt}, shape: {full_kpt.shape}")
                # print(f"position: {position}/n, Full_kp: {keypoint_tensor}, shape: {keypoint_tensor.shape}")

                sliding_window[position].append((crop_tensor, keypoint_tensor))
                # print(f"sliding_window {sliding_window[position]}, length: {len(sliding_window[position])}")

                if len(sliding_window[position]) == WINDOW_SIZE:
                    crops_stack = torch.stack([x[0] for x in sliding_window[position]], dim=1)  # (1, 5, 1, 64, 64)
                    # keypoint_input = sliding_window[position][-1][1]  # Most recent (1, 101)
                    kpt_list = [x[1] for x in sliding_window[position]]
                    # kpts_stack = torch.stack(kpt_list, dim=0)  # (5, 101)
                    # keypoint_input = kpts_stack.view(1, -1)  # (1, 505)
                    kpts_tensor = torch.cat(kpt_list, dim=1)  # (5, 101)
                    position_tensor = torch.tensor([[roi['position']]], dtype=torch.float32, device=DEVICE)
                    keypoint_input = torch.cat([kpts_tensor, position_tensor], dim=1)

                    with torch.no_grad():
                        output = fusion_model(crops_stack, keypoint_input)
                        pred = output.argmax(dim=1).item()
                        label = "HAND IN POCKET" if pred == 1 else "NO HAND"
                        color = (0, 0, 255) if pred == 1 else (0, 255, 0)

                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(display_frame, f"Desk {roi['desk']} - {label}",
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("3D CNN Hybrid Inference", display_frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cap.release()
    # cv2.destroyWindow(f"desk_{desk_num}")
cv2.destroyAllWindows()
