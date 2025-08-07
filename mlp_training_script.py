import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -------- Configuration --------
INPUT_SIZE = 104
HIDDEN_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 5
# MODEL_PATH = "mlp_hand_in_pocket.pt"
model_name = "mlp_temp_regrouped_pos_gen_round-c0"
label_column = 'hand_in_pocket'  

# -------- Load Dataset --------
csv_name = 'new_combine_round.csv' 
# "C:/wajahat/hand_in_pocket/dataset/without_kp/
df = pd.read_csv(f'C:/wajahat/hand_in_pocket/dataset/without_kp/{csv_name}')  # Replace with your actual path

df = df.drop(columns=['camera', 'video', 'frame', 'desk']) 
# df = df.drop(columns=['source_file']) 

# Handle empty strings as missing
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(-1, inplace=True)  # Set missing keypoints to -1

# Separate features and labels
X = df.drop(columns=[label_column]).values.astype(np.float32)  # Adjust if label column is different
y = df[label_column].values.astype(np.float32)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------- MLP Model --------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            # nn.Linear(104, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            nn.Linear(104, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MLP(INPUT_SIZE, HIDDEN_SIZE)

# -------- Training Setup --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

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
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device).unsqueeze(1)
            outputs = model(val_X)
            loss = criterion(outputs, val_y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"rf_models/{model_name}.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"Training complete. Best model saved to rf_models/{model_name}.pt")

# def compute_feature_importance(model, dataloader):
#     model.eval()
#     importance_list = []

#     for batch_X, _ in dataloader:
#         batch_X = batch_X.to(device)
#         batch_X.requires_grad = True

#         outputs = model(batch_X)
#         model.zero_grad()
#         outputs.sum().backward()  # sum() to backprop all outputs

#         grads = batch_X.grad.abs().detach().cpu().numpy()
#         importance_list.append(grads)

#     all_grads = np.concatenate(importance_list, axis=0)
#     feature_importance = np.mean(all_grads, axis=0)
#     return feature_importance

# # Load best model and compute importance
# model.load_state_dict(torch.load(f"rf_models/{model_name}.pt"))
# feature_importance = compute_feature_importance(model, val_loader)

# # Save to text file
# with open(f"{model_name}.txt", "w") as f:
#     f.write("Feature Index\tImportance Score\n")
#     for idx, score in enumerate(feature_importance):
#         f.write(f"{idx}\t{score:.6f}\n")

# print(f"Feature importance saved to {model_name}")
