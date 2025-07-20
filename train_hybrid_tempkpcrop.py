"""Training script for hybrid temporal keypoint and crop feature extraction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from dataloader_hybrid_tempkpcrop import TemporalKeypointCropDataset
from dataloader_hybrid_pos_gen import TemporalKeypointCropDataset
# from model_hybrid_tempkpcrop import FusionModel
from model_3dcnn_hybrid import Fusion3DCNNKeypointModel as FusionModel # 3D CNN model architecture

def train_model(csv_path, crop_root, epochs, batch_size, model_name, lr=1e-3, patience=5):
    dataset = TemporalKeypointCropDataset(csv_path, crop_root)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = FusionModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for crops, keypoints, labels in train_loader:
            crops, keypoints, labels = crops.cuda(), keypoints.cuda(), labels.cuda()
            outputs = model(crops, keypoints)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for crops, keypoints, labels in val_loader:
                crops, keypoints, labels = crops.cuda(), keypoints.cuda(), labels.cuda()
                outputs = model(crops, keypoints)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"rf_models/{model_name}")
            print("✅ Saved new best model")
        else:
            patience_counter += 1
            print(f"⏳ Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break


csv_path = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/cnn_combine_norm_pos_gen.csv"
crop_root = "C:/wajahat/hand_in_pocket/dataset/scheck/without_kp_crop"
epochs = 50
batch_size = 16
model_name = "hybrid_tempkpcrop_norm.pth"
train_model(csv_path, crop_root, epochs, batch_size, model_name, lr=1e-3, patience=5)