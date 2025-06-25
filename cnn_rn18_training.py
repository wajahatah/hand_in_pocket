""" Train a CNN model using ResNet18 architecture on a custom dataset. 
It includes data augmentation, use one and two neurons in the final layer respectively,
and implements early stopping based on validation loss."""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import copy

# ========== CONFIG ==========
data_dir = "C:/wajahat/hand_in_pocket/dataset/without_kp"
batch_size = 8
img_size = 640
num_epochs = 300
patience = 10  # Early stopping
model_name = "cnn_rn_arg_aug.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DATA TRANSFORM ==========
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# ========== LOAD DATASET ==========
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# ========== CREATE TRAIN/VAL SPLIT ==========
targets = np.array([s[1] for s in full_dataset.samples])
train_idx, val_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

# ========== APPLY VAL TRANSFORM TO VALIDATION SET ==========
full_dataset_val = datasets.ImageFolder(root=data_dir, transform=val_transform)

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(full_dataset_val, batch_size=batch_size, sampler=val_sampler)

# ========== LOAD PRETRAINED MODEL ==========
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for binary classification
num_features = model.fc.in_features

# to return both classes (hand in pocket and no hand in pocket)
model.fc = nn.Linear(num_features, 2)

# use only one neuron in the final layer
# model.fc = nn.Linear(num_features, 1)

model = model.to(device)

# ========== LOSS + OPTIMIZER ==========
# for two neurons in the final layer
criterion = nn.CrossEntropyLoss()

# for one neuron in the final layer
# criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

# ========== EARLY STOPPING VARIABLES ==========
best_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

# ========== TRAINING LOOP ==========
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # ---------- Training ----------
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # for two neurons in the final layer
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # for one neuron in the final layer
        # optimizer.zero_grad()
        # outputs = model(images).squeeze(1)
        # labels = labels.float()  
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # running_loss += loss.item() * labels.size(0)
        # preds = (torch.sigmoid(outputs) > 0.5).long()
        # correct += (preds == labels).sum().item()
        # total += labels.size(0)

    train_loss = running_loss / len(train_sampler)
    train_acc = correct / total * 100

    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # for two neurons in the final layer
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # for one neuron in the final layer
            # outputs = model(images).squeeze(1)
            # labels = labels.float()
            # loss = criterion(outputs, labels)
            # val_loss += loss.item() * labels.size(0)

            # preds = (torch.sigmoid(outputs) > 0.5).long()
            # correct += (preds == labels).sum().item()
            # total += labels.size(0)

    val_loss /= len(val_sampler)
    val_acc = correct / total * 100

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    # ---------- Early Stopping ----------
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, f"rf_models/{model_name}")
        print("✅ Validation loss decreased, saving model...")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print("\n⛔ Early stopping triggered.")
            break

# ========== LOAD BEST MODEL ==========
model.load_state_dict(best_model_wts)
print(f"\n🏆 Best model loaded from '{model_name}'")
