import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# ======== Hyperparameters ========
INPUT_SIZE = 24         # feature_dim per timestep
WINDOW_SIZE = 5         # timesteps
HIDDEN_SIZE = 64
NUM_LAYERS = 3
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 10
MODEL_PATH = "gru_temp_pos_gen"
csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_l1_v2_norm_pos_gen_rnn.csv"

# ========== GRU Model ==========
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # use last timestep
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(1)

# ========== Dataset ==========
class HandInPocketDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        df = df.drop(columns=['source_file'])
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna()  
        df = df.apply(pd.to_numeric)
        data = df.values.astype(np.float32)
        self.features = data[:, :-1].reshape(-1, WINDOW_SIZE, INPUT_SIZE)
        self.labels = data[:, -1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# ========== Early Stopping ==========
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ========== Training ==========
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HandInPocketDataset(csv)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        early_stopping(val_loss, model, f"rf_models/{MODEL_PATH}.pth")
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Best model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
