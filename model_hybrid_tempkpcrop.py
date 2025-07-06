""" Model architecture for hybrid temporal keypoint and crop feature extraction.
   - CropCNN: Convolutional Neural Network for image crops.
   - KeypointMLP: Multi-Layer Perceptron for keypoint features.
   - FusionModel: Combines features from both CNN and MLP, followed by a classifier."""

import torch
import torch.nn as nn

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