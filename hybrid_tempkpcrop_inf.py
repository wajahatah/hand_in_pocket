import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from collections import deque
from ultralytics import YOLO
# from model_hybrid_tempkpcrop import FusionModel  # Make sure this is accessible
# from model_3dcnn_hybrid import Fusion3DCNNKeypointModel as FusionModel  # 3D CNN model architecture
import torch.nn as nn

# ========= CONFIG =========
YOLO_MODEL_PATH = "bestv7-2.pt"
FUSION_MODEL_PATH = "rf_models/hybrid_tempkpcrop_model-1.pth"
JSON_PATH = "qiyas_multicam.camera_final.json"
VIDEO_DIR = "C:/wajahat/hand_in_pocket/test_bench"
# VIDEO_DIR = "F:/Wajahat/hand_in_pocket/qiyas_test"
CROP_SIZE = (64, 64)
WINDOW_SIZE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

global color, label
color = (0,0,0)

# ======== Model Architecture =========
# """
class CropCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        # self.fc = nn.Linear(32 * 16 * 16, out_dim)
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return x.view(B, T, -1)

class KeypointMLP(nn.Module):
    def __init__(self, in_dim=101, out_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):  # x: (B, 101)
        return self.model(x)

class FusionModel(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.cnn = CropCNN(out_dim=feat_dim)
        self.kpt_mlp = KeypointMLP(in_dim=101, out_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 6, 256),
            nn.ReLU(), 
            # nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, crops, keypoints):
        crop_feat = self.cnn(crops).view(crops.size(0), -1)  # (B, 5*128)
        kpt_feat = self.kpt_mlp(keypoints)  # (B, 128)
        fused = torch.cat([crop_feat, kpt_feat], dim=-1)  # (B, 768)
        return self.classifier(fused)
# """

# ========= LOAD MODELS =========
print("Loading YOLO keypoint model...")
kp_model = YOLO(YOLO_MODEL_PATH).to(DEVICE)

print("Loading FusionModel...")
fusion_model = FusionModel().to(DEVICE)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
fusion_model.eval()

# ========= UTILS =========
def get_position(x, roi_data_list):
    for roi in roi_data_list:
        if roi['xmin'] <= x <= roi['xmax']:
            return roi['position']
    return None

def extract_keypoints(kp_tensor):
    flat = []
    for kp in kp_tensor[:10]:  # 10 keypoints
        x, y, conf = kp[:3].cpu().numpy()
        if conf < 0.5:
            x, y = -1, -1
        flat.extend([x, y])
    return flat

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
        cam_id = input("Enter camera ID for this video: ")
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
    desk_data = camera_data['data']
    # desk_num = roi['desk']

    print(f"Data: {roi_data_list}")
    roi_lookup = {roi['position']: roi for roi in roi_data_list}
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
            for person_idx, kp_tensor in enumerate(keypoint_tensor):
                keypoints = []
                keypoints_c = []
                feature_vector = []
                for i, keypoint in enumerate(kp_tensor[:10]):
                    x, y, conf = keypoint[:3].cpu().numpy()
                    if conf < 0.5:
                        x, y = 0, 0

                    keypoints_c.append((x, y))
                    keypoints.extend([x, y])

                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.circle(display_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                if len(keypoints_c) == 0 or all((x == 0 and y == 0) for x, y in keypoints_c):
                    continue

                # Estimate center X for desk matching
                person_x = keypoints_c[0][0]

                position = get_position(person_x, roi_data_list)
                if position is None:
                    continue

                roi = roi_lookup.get(position)
                if not roi:
                    continue

                xmin, ymin, xmax, ymax = roi['xmin'], roi['ymin'], roi['xmax'], roi['ymax']
                # print(f"position: {position}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
                crop = frame[:, xmin:xmax]

                desk_num = roi['desk']
                # for desk_id, roi in desk_data.items():
                #     xmin, xmax = roi['xmin'], roi['xmax']
                #     ymin, ymax = roi['ymin'], roi['ymax']
                    # print(f"desk: {desk_num}, xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

                    # crop = frame[:, xmin:xmax]
                # cv2.imshow(f"crop_desk_{desk_num}", crop)

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, CROP_SIZE)
                norm_crop = gray_resized.astype(np.float32) / 255.0
                crop_tensor = torch.tensor(norm_crop, device=DEVICE).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)

                keypoint_tensor = torch.tensor(keypoints, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Fill sliding window
                sliding_window[position].append((crop_tensor, keypoint_tensor))

                if len(sliding_window[position]) == WINDOW_SIZE:
                    # FIX: Properly stack the crops to get (1, 5, 1, 64, 64) shape
                    crops_list = [x[0] for x in sliding_window[position]]  # List of (1,1,64,64) tensors
                    crops_stack = torch.stack(crops_list, dim=1)  # Stack along time dimension: (1, 5, 1, 64, 64)
                    # crops_stack is already on the correct device since individual tensors are on device
                    
                    # FIX: Properly handle keypoints tensor
                    kpts_list = [x[1] for x in sliding_window[position]]  # List of (1,20) tensors
                    kpts_tensor = torch.stack(kpts_list, dim=0)  # (5, 1, 20)
                    kpts_tensor = kpts_tensor.view(-1)  # Flatten to (100,)
                    
                    # Add position value to make it 101 dimensions
                    position_tensor = torch.tensor([roi['position']], dtype=torch.float32, device=DEVICE)
                    keypoints_with_pos = torch.cat([kpts_tensor, position_tensor]).unsqueeze(0)  # (1,101)

                    # Debug prints to verify shapes
                    # print(f"crops_stack shape: {crops_stack.shape}")  # Should be (1, 5, 1, 64, 64)
                    # print(f"keypoints_with_pos shape: {keypoints_with_pos.shape}")  # Should be (1, 101)
                    # print(f"crops_stack device: {crops_stack.device}")
                    # print(f"keypoints_with_pos device: {keypoints_with_pos.device}")
                    # print(f"position: {position}, keypoints_with_pos: {keypoints_with_pos}, crops_stack: {crops_stack}")

                    with torch.no_grad():
                        output = fusion_model(crops_stack, keypoints_with_pos)
                        pred = output.argmax(dim=1).item()
                        label = "HAND IN POCKET" if pred == 1 else "NO HAND"
                        color = (0, 0, 255) if pred == 1 else (0, 255, 0)

                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(display_frame, f"Desk {roi['desk']} - {label}",
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Hybrid Inference", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
cv2.destroyAllWindows()