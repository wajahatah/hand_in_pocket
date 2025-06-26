import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================== CONFIG ====================
data_root = "C:/wajahat/hand_in_pocket/dataset/without_kp_crop"
temporal_window = 5
img_size = 640
batch_size = 8
epochs = 300
model_name = "cnn_temp_gray_crop.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DATASET ====================
class TemporalGrayDataset(Dataset):
    def __init__(self, image_paths, label, window_size=5):
        self.samples = []
        self.label = label
        self.window_size = window_size

        grouped = {}
        for path in image_paths:
            # name = os.path.basename(path)
            # key_parts = name.split("_")
            # group_key = "_".join(key_parts[:3])  # e.g., c1_v1_desk2
            # frame_num = int(key_parts[3][1:5])   # from f0003.jpg → 3
            # grouped.setdefault(group_key, []).append((frame_num, path))

            name = os.path.basename(path)
            key_parts = name.split("_")
            if len(key_parts) < 4:
                continue  # skip malformed names

            group_key = "_".join(key_parts[:3])  # e.g., 'c1_v1_d1'

            try:
                frame_str = key_parts[3].split(".")[0]  # 'f0000'
                frame_num = int(frame_str[1:])  # strip 'f', get number
            except ValueError:
                continue  # skip bad formats

            grouped.setdefault(group_key, []).append((frame_num, path))

        for group_key in grouped:
            frames = sorted(grouped[group_key])
            for i in range(len(frames) - window_size + 1):
                clip = [f[1] for f in frames[i:i+window_size]]
                self.samples.append(clip)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        clip = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            clip.append(img)
        clip_tensor = torch.tensor(clip).unsqueeze(1)  # shape: (5, 1, 640, 640)
        clip_tensor = clip_tensor.squeeze(1)  # (5, 640, 640)
        label_tensor = torch.tensor(self.label, dtype=torch.float32)
        return clip_tensor, label_tensor

# ==================== MODEL ====================
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
            nn.Linear(16 * 5 * 160 * 160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, 5, 640, 640)
        x = x.unsqueeze(1)  # (B, 1, 5, 640, 640)
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze()

# ==================== PREPARE DATA ====================
hp_paths = glob(os.path.join(data_root, "hand_in_pocket", "*.jpg"))
nop_paths = glob(os.path.join(data_root, "no_hand_in_pocket", "*.jpg"))

hp_train, hp_val = train_test_split(hp_paths, test_size=0.2, random_state=42)
nop_train, nop_val = train_test_split(nop_paths, test_size=0.2, random_state=42)

train_dataset = TemporalGrayDataset(hp_train, 1) + TemporalGrayDataset(nop_train, 0)
val_dataset = TemporalGrayDataset(hp_val, 1) + TemporalGrayDataset(nop_val, 0)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==================== TRAIN ====================
model = TemporalCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = float('inf')
early_stop_counter = 0
patience = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for clips, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_temporal_model.pt")
        early_stop_counter = 0
        print("Saved new best model.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
