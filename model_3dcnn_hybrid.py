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
    def __init__(self, in_dim=104, out_dim=32):
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
        self.kpt = KeypointMLP(in_dim=104, out_dim=kpt_feat_dim)

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
