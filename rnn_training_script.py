import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# -------------------- Dataset Class --------------------
class HandInPocketDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        df = df.drop(columns=['source_file']) 
        # self.source_files = df['source_file'].values

        df = df.replace(r'^\s*$', pd.NA, regex=True)  # Replace empty strings with NaN
        df = df.dropna()  # Drop rows with NaN values
        df = df.apply(pd.to_numeric)

        self.labels = df['label'].values.astype(np.float32)
        # self.features = df.drop(columns=['source_file','label']).values.astype(np.float32)
        self.features = df.drop(columns=['label']).values.astype(np.float32)
        self.features[self.features == -1] = -10.0

        assert not np.isnan(self.features).any(), "❌ NaNs found in features"
        assert not np.isnan(self.labels).any(), "❌ NaNs found in labels"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx].reshape(5, 24)  # 5 time steps, 24 features per step
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -------------------- RNN Model --------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # take output of last time step
        out = self.fc(out)
        # return self.sigmoid(out).squeeze(1)
        return out.squeeze(1)

# -------------------- Training Loop --------------------
# def train_model(csv_path, epochs=50, batch_size=32, lr=0.001, patience=5, model_save_path='best_rnn_model.pth'):
def train_model(csv_path, epochs, batch_size, lr, patience, model_save_path):
    dataset = HandInPocketDataset(csv_path)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    # train_val_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)  # 16% val

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    # test_set = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    # test_loader = DataLoader(test_set, batch_size=batch_size)

    model = RNNClassifier()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x = torch.clamp(x, min=-10.0, max=1.0)

            preds = model(x)
            loss = criterion(preds, y)
            if torch.isnan(loss):
                print("❌ NaN loss encountered, skipping this batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = torch.clamp(x, min=-10.0, max=1.0)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
                # predicted = (preds > 0.5).float()
                predicted = (torch.sigmoid(preds) > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader.dataset)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}")

        # Early Stopping and Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("✅ Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(torch.load(model_save_path))

    # Test Evaluation
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         preds = model(x)
    #         predicted = (preds > 0.5).float()
    #         correct += (predicted == y).sum().item()
    #         total += y.size(0)

    # test_accuracy = correct / total
    # print(f"\n🧪 Final Test Accuracy: {test_accuracy:.4f}")

    return model

# Example usage:
# model = train_model('hand_in_pocket_rnn_ready.csv')


csv_path = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_l1_v2_norm_pos_gen_rnn.csv"
epochs = 300
batch_size = 32
lr = 0.001
model_name = "rnn_norm_pos_gen-c0.pth"
# model_save_path = "best_rnn_model.pth"
model_save_path = f"rf_models/{model_name}"
patience = 10

train_model(csv_path, epochs, batch_size, lr, patience, model_save_path)
print(f"✅ Model trained and saved at: {model_save_path}")
