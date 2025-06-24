import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import copy

# ========== CONFIG ==========
data_dir = "C:/wajahat/hand_in_pocket/dataset/without_kp"
batch_size = 16
img_size = 640
num_epochs = 300
patience = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "cnn_custom.pth"

# ========== TRANSFORMS ==========
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# ========== LOAD DATA ==========
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

targets = np.array([s[1] for s in train_dataset.samples])
train_idx, val_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# ========== CUSTOM CNN MODEL ==========
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 640x640 → 640x640
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 320x320

            nn.Conv2d(16, 32, 3, padding=1),  # → 320x320
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 160x160

            nn.Conv2d(32, 64, 3, padding=1),  # → 160x160
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 80x80

            nn.Conv2d(64, 128, 3, padding=1),  # → 80x80
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 40x40

            nn.Flatten(),
            nn.Linear(128 * 40 * 40, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # single output for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = CustomCNN().to(device)

# ========== LOSS + OPTIMIZER ==========
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ========== EARLY STOPPING INIT ==========
best_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
no_improve_epochs = 0

# ========== TRAINING LOOP ==========
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = correct / total * 100

    # ---------- VALIDATION ----------
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total * 100

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    # ---------- EARLY STOPPING ----------
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, f"rf_models/{model_name}")
        print("✅ Validation loss improved. Model saved.")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement for {no_improve_epochs} epoch(s)")

        if no_improve_epochs >= patience:
            print("⛔ Early stopping triggered.")
            break

# ========== LOAD BEST MODEL ==========
model.load_state_dict(best_model_wts)
print("\n🏆 Best model loaded: cnn_best.pth")
