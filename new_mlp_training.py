import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# -------- Configuration --------
INPUT_SIZE = 64
# HIDDEN_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 25
model_name = "mlp_temp_balanced_norm-wajahat-t3-c0"
label_column = 'hand_in_pocket'

# -------- Load Dataset --------
df = pd.read_csv("C:/wajahat/hand_in_pocket/dataset/training2/new_combined_temp_balanced_norm.csv")
df = df.drop(columns=['camera', 'video', 'frame', 'desk_no',
                      'kp_0_x_t1', 'kp_0_x_t3', 'kp_0_y_t1', 'kp_0_y_t3',
                        'kp_1_x_t1', 'kp_1_x_t3', 'kp_1_y_t1', 'kp_1_y_t3',
                        'kp_2_x_t1', 'kp_2_x_t3', 'kp_2_y_t1', 'kp_2_y_t3',
                        'kp_3_x_t1', 'kp_3_x_t3', 'kp_3_y_t1', 'kp_3_y_t3',
                        'kp_4_x_t1', 'kp_4_x_t3', 'kp_4_y_t1', 'kp_4_y_t3',
                        'kp_5_x_t1', 'kp_5_x_t3', 'kp_5_y_t1', 'kp_5_y_t3',
                        'kp_6_x_t1', 'kp_6_x_t3', 'kp_6_y_t1', 'kp_6_y_t3',
                        'kp_7_x_t1', 'kp_7_x_t3', 'kp_7_y_t1', 'kp_7_y_t3',
                        'kp_8_x_t1', 'kp_8_x_t3', 'kp_8_y_t1', 'kp_8_y_t3',
                        'kp_9_x_t1', 'kp_9_x_t3', 'kp_9_y_t1', 'kp_9_y_t3'])

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(-1, inplace=True)

X = df.drop(columns=[label_column]).values.astype(np.float32)
y = df[label_column].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------- MLP Model --------
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(INPUT_SIZE)

# -------- Training Setup --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Create save directory
os.makedirs("rf_models", exist_ok=True)

# -------- Early Stopping --------
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device).unsqueeze(1)
            outputs = model(val_X)
            loss = criterion(outputs, val_y)
            val_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            val_accuracy += (predictions == val_y).float().mean().item()
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {avg_val_accuracy:.4f}")

    # Early stopping and model saving
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, f"rf_models/{model_name}.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    scheduler.step(avg_val_loss)

print(f"Training complete. Best model saved to rf_models/{model_name}.pt")