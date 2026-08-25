import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

DATA_DIR = os.path.abspath("./data/balanced")
RESULTS_DIR = os.path.abspath("./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Combine a few datasets for a robust ablation study
DATA_FILES = ["chapman_4_classes_balanced.npz", "cpsc_clean.npz", "georgia_clean.npz"]

# --- 1. Model ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=9, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.drop(F.relu(self.bn(self.conv(x))))

class ResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        w = F.softmax(self.attention(x).squeeze(-1), dim=1)
        return torch.sum(w.unsqueeze(-1) * x, dim=1), w

class CNN1DAttentionEnhancedLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock(12, 16), ResidualUnit(16), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(ConvBlock(16, 32), ResidualUnit(32), nn.MaxPool1d(2))
        self.block3 = nn.Sequential(ConvBlock(32, 64), ResidualUnit(64), nn.MaxPool1d(2))
        self.block4 = nn.Sequential(ConvBlock(64, 128, kernel_size=7), ResidualUnit(128), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(128, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = Attention(32)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.block4(self.block3(self.block2(self.block1(x)))).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

print("Loading subset of data for ablation...")
X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []

for filename in DATA_FILES:
    d = np.load(os.path.join(DATA_DIR, filename))
    X_train_list.append(torch.tensor(d['X_train'], dtype=torch.float32))
    y_train_list.append(torch.tensor(d['y_train'], dtype=torch.long))
    X_val_list.append(torch.tensor(d['X_val'], dtype=torch.float32))
    y_val_list.append(torch.tensor(d['y_val'], dtype=torch.long))

X_train = torch.cat(X_train_list)
y_train = torch.cat(y_train_list)
X_val = torch.cat(X_val_list)
y_val = torch.cat(y_val_list)
print(f"Total ablation training samples: {len(X_train)}")

# Define Hyperparameters to test
learning_rates = [0.001, 0.002, 0.003]
batch_sizes = [32, 64, 128]
epochs = 5  # Reduced epochs for faster ablation grid search

results = []

for bs in batch_sizes:
    for lr in learning_rates:
        print(f"\n--- Testing LR={lr}, BatchSize={bs} ---")
        model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), bs):
                idx = perm[i:i+bs]
                optimizer.zero_grad()
                loss = criterion(model(X_train[idx].to(DEVICE)), y_train[idx].to(DEVICE))
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        preds, true_y = [], []
        with torch.no_grad():
            for i in range(0, len(X_val), bs):
                bx = X_val[i:i+bs].to(DEVICE)
                by = y_val[i:i+bs]
                out = model(bx)
                preds.extend(torch.max(out, 1)[1].cpu().numpy())
                true_y.extend(by.numpy())
        
        wf1 = f1_score(true_y, preds, average='weighted', zero_division=0)
        print(f"Result -> LR={lr}, BS={bs}, Val F1={wf1:.4f}")
        results.append({"LR": lr, "BatchSize": bs, "Val_F1": wf1})

df = pd.DataFrame(results)
out_csv = os.path.join(RESULTS_DIR, "table8_lr_bs_ablation.csv")
df.to_csv(out_csv, index=False)
print(f"\nAblation completed! Results saved to {out_csv}")
