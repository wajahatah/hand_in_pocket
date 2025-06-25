""" load the frames convert them into gray scale, stack them into the window of 5 for the temporal feature, train the custom model"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import copy

# ======== CONFIG ========
data_root = "C:/wajahat/hand_in_pocket/dataset/without_kp"
batch_size = 16
img_size = 640
sequence_length = 5
num_epochs = 300
patience = 5
model_name = "cnn_temp_gray.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== CUSTOM DATASET ========
def gen_sequence(image_paths, labels, sequence_length):
    sequences = []
    for i in range(len(image_paths) - sequence_length + 1):
        seq = image_paths[i:i + sequence_length]
        sequences.append((seq, labels))
    return sequences

hp = sorted(glob.glob(os.path.join(data_root, 'hp', '*.jpg')))
no_hp = sorted(glob.glob(os.path.join(data_root, 'no_hp', '*.jpg')))

hand_sequences = gen_sequence(hp, 1, sequence_length)
no_hand_sequences = gen_sequence(no_hp, 0, sequence_length)

all_sequences = hand_sequences + no_hand_sequences
random.shuffle(all_sequences)

train_seqs, val_seqs = train_test_split( all_sequences, test_size=0.2, random_state=42, stratify=[label for _, label in all_sequences])

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        frame_paths, label = self.sequences[idx]
        frames = []
        for path in frame_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0 
            frames.append(img)
        stacked = np.stack(frames, axis=0)  # Shape: (sequence_length, img_size, img_size)
        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
train_dataset = SequenceDataset(train_seqs)
val_dataset = SequenceDataset(val_seqs)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ======== CUSTOM CNN ========
class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),  # input channels = 5 (grayscale stack)
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
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(1)  # [B]


# ======== TRAINING SETUP ========
model = TemporalCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
no_improve_epochs = 0

# ======== TRAINING LOOP ========
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # ---- Train ----
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = 100 * correct / total

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = 100 * correct / total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    # ---- Early Stopping ----
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, f"rf_models/{model_name}")
        print(f"✅ Saved best {model_name}.")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement for {no_improve_epochs} epoch(s)")
        if no_improve_epochs >= patience:
            print("⛔ Early stopping triggered.")
            break

# ======== DONE ========
print("🏁 Training complete. Loading best model.")
model.load_state_dict(best_model_wts)
