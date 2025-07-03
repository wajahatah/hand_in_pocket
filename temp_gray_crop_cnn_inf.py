import os
import cv2
import json
import torch
import numpy as np
from collections import deque
from tqdm import tqdm
import torch.nn as nn
from ultralytics import YOLO


# ============ CONFIG ============
# video_path = "C:/wajahat/hand_in_pocket/test_bench/fp_t1.mp4"
input_path = "C:/wajahat/hand_in_pocket/test_bench/"
json_path = "qiyas_multicam.camera_final.json"
model_path = "C:/wajahat/hand_in_pocket/rf_models/cnn_temp_gray_crop_withoutkp-1.pth"
temporal_window = 5
img_size = 640
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kp_model = YOLO("bestv7-2.pt")
keypoints = False  

# ============ MODEL DEFINITION ============
class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 160 * 160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, 5, 640, 640)
        x = x.unsqueeze(1)  # (B, 1, 5, 640, 640)
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze()

video_files = [f for f in os.listdir(os.path.dirname(input_path)) if f.endswith('.mp4')]
if not video_files:
    print("No video files found in the specified directory.")
    exit()

for video_file in video_files:
    video_path = os.path.join(input_path, video_file)
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video.")

    first_frame = cv2.resize(first_frame, (1280, 720))  # Resize for better visibility
    cv2.imshow("First Frame", first_frame)
    cv2.waitKey(1)

    with open(json_path, 'r') as f:
        camera_configs = json.load(f)

    skip_video = False
    while True:
        cam_id = input("Enter camera ID: ")
        if cam_id.lower() == 's':
            skip_video = True
            cap.release()
            cv2.destroyWindow("Select Camera")
            break
        cam_key = f"camera_{cam_id}"
        camera_data = next((cam for cam in camera_configs if cam["_id"] == cam_key), None)
        if camera_data:
            break
        print("Invalid camera ID. Try again.")

    if skip_video:
        continue

    # camera_ids = [cam['_id'] for cam in camera_configs]
    # print(f"Available Cameras: {camera_ids}")
    # camera_input = input("Enter camera ID (e.g., camera_1): ").strip()

    # # camera_data = next((c for c in camera_configs if c['_id'] == camera_input), None)
    # camera_data = next((c for c in camera_configs if c['_id'] == f"camera_{camera_input}"), None)
    # if not camera_data:
    #     raise ValueError(f"No config found for camera {camera_input}")

    cv2.destroyAllWindows()

    desk_data = camera_data["data"]

    # ============ LOAD MODEL ============
    model = TemporalCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ============ PREPARE BUFFERS ============
    desk_buffers = {desk_id: deque(maxlen=temporal_window) for desk_id in desk_data.keys()}
    desk_predictions = {desk_id: "..." for desk_id in desk_data.keys()}

    # ============ READ VIDEO ============

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        display_frame = frame.copy()
        display_frame = cv2.resize(display_frame, (1280, 720))  # Resize for better visibility

        if keypoints == True:
            results = kp_model(frame)
            for result in results:
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue

                keypoints_tensor = result.keypoints.data
                for person_idx, kp_tensor in enumerate(keypoints_tensor):
                    for i, keypoint in enumerate(kp_tensor[:10]):
                        x, y, conf = keypoint[:3].cpu().numpy()
                        if conf > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            cv2.circle(display_frame, (int(x), int(y)), 5, (0, 255, 0), -1)


        for desk_id, roi in desk_data.items():
            xmin, xmax = roi['xmin'], roi['xmax']
            ymin, ymax = roi['ymin'], roi['ymax']
            desk_num = roi['desk']

            # Draw ROI box
            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Crop and preprocess for model
            crop = frame[:, xmin:xmax]  # cropped area for model
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            normed = resized.astype(np.float32) / 255.0
            desk_buffers[desk_id].append(normed)

            # cv2.imshow(f"crop_desk_{desk_num}", crop)
            # Run model if enough frames in buffer
            if len(desk_buffers[desk_id]) == temporal_window:
                clip = np.array(desk_buffers[desk_id])
                clip_tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)
                # print(f"clip tensor: {clip_tensor}")
                # cv2.imshow("clip", clip_tensor)
                # print(f"desk_num: {desk_num}, clip_tensor: {clip_tensor}")
                with torch.no_grad():
                    pred = model(clip_tensor).item()
                    label = "HAND IN POCKET" if pred > 0.5 else "NO HAND"
                    desk_predictions[desk_id] = f"Desk {desk_num}: {label} ({pred:.2f})"

            # Show label on frame
                    color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
                    label_text = desk_predictions[desk_id]
                    cv2.putText(display_frame, label_text, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # cv2.putText(crop, label_text, (xmin, ymin - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Inference", display_frame)
        # cv2.imshow("Inference", frame)
        # cv2.imshow("Inference", crop)

        # cv2.imshow("Inference", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
cv2.destroyAllWindows()
