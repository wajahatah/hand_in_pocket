import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import os


# ========== PREPROCESSING FUNCTION ==========
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [H, W]
    gray = cv2.resize(gray, (img_size, img_size))   # resize
    gray = gray.astype(np.float32) / 255.0          # normalize
    return gray  # shape: [640, 640]

# ========== TEMPORAL CNN MODEL ==========
class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 40 * 40, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(1)
    
# ========== CONFIG ==========
video_dir = "C:/wajahat/hand_in_pocket/test_bench"
model_path = "C:/wajahat/hand_in_pocket/rf_models/cnn_temp_gray.pth"
img_size = 640
sequence_length = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_files = [ f for f in os.listdir(video_dir) if f.endswith('.mp4') ]
if not video_files:
    print("No video files found in the directory.")
    exit()

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing video: {video_path}")
#"F:/Wajahat/hand_in_pocket/hand_in_pocket/cam_1/chunk_26-02-25_10-15-desk1-2-3.avi"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        continue

    ret, preview_frame = cap.read()
    preview_frame = cv2.resize(preview_frame, (1280, 720))
    cv2.imshow("Select Camera",  preview_frame)
    cv2.waitKey(1)
    skip_video = False
    while True:
        camera_id_input = input("Enter camera ID: ")
            
        if camera_id_input.lower() == 's':
            print(f"Skipping video {video_path}")
            # cap.release()
            # cv2.destroyWindow("Select Camera")
            skip_video = True
            break

        else:
            break

    cap.release()
    cv2.destroyWindow("Select Camera")

    if skip_video == True:
        continue

    else: 


    # ========== LOAD MODEL ==========
        model = TemporalCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # ========== VIDEO STREAM ==========
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Could not open video.")
            exit()

        frame_buffer = []
        print("🔍 Running temporal inference... (press 'q' to quit)\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame,(1280,720))
            # --- Preprocess and add to buffer ---
            gray = preprocess_frame(frame)
            frame_buffer.append(gray)

            if len(frame_buffer) < sequence_length:
                continue  # not enough frames to infer

            if len(frame_buffer) > sequence_length:
                frame_buffer.pop(0)

            # --- Stack into tensor [5, 640, 640] ---
            input_tensor = np.stack(frame_buffer, axis=0)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 5, 640, 640]

            with torch.no_grad():
                logit = model(input_tensor)
                prob = torch.sigmoid(logit).item()
                label = "HAND IN POCKET" if prob > 0.5 else "NO HAND IN POCKET"

            # --- Display result ---
            text = f"{label} ({prob:.2f})"
            if label == "HAND IN POCKET":
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.imshow("Temporal CNN Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
print("✅ Inference finished.")
