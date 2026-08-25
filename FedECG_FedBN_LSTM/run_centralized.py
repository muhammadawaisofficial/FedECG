"""
run_centralized.py — Centralized Learning Baseline (Table 3)
Combines ALL 6 hospital datasets into one large training set and trains 
a single CNN-LSTM-Attention model WITHOUT any federated aggregation.
This serves as the theoretical upper-bound for the federated methods.
"""
import os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Centralized Learning Baseline | Device: {DEVICE}")

DATA_DIR = os.path.abspath("./data/balanced")
RESULTS_DIR = os.path.abspath("./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_FILES = {
    "Chapman": "chapman_4_classes_balanced.npz",
    "CPSC": "cpsc_clean.npz",
    "Georgia": "georgia_clean.npz",
    "Ningbo": "ningbo_clean.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx): return torch.tensor(self.signals[idx]), torch.tensor(self.labels[idx])

# --- Model Architecture (same as federated) ---
class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1), weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
    def forward(self, x):
        return self.dropout(F.leaky_relu(self.bn(self.conv(x))).unsqueeze(2)).squeeze(2)

class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(channels, channels, 9, padding=4), nn.BatchNorm1d(channels),
                                   nn.LeakyReLU(), nn.Conv1d(channels, channels, 9, padding=4), nn.BatchNorm1d(channels))
    def forward(self, x): return x + self.block(x)

class CNN1DAttentionEnhancedLSTM(nn.Module):
    def __init__(self, input_length=3100, num_classes=3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(12, 64), ResidualUnit(64), nn.MaxPool1d(4),
            ConvBlock(64, 128), ResidualUnit(128), nn.MaxPool1d(4),
            ConvBlock(128, 256), ResidualUnit(256), nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention(256)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv_blocks(x).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x)
        return self.classifier(x)

if __name__ == "__main__":
    # 1. Load and combine ALL datasets
    print("📦 Loading and combining all 6 datasets...")
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    X_test_all, y_test_all = [], []

    for name, fname in DATA_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        d = np.load(path)
        X_train_all.append(d['X_train']); y_train_all.append(d['y_train'])
        X_val_all.append(d['X_val']);     y_val_all.append(d['y_val'])
        X_test_all.append(d['X_test']);   y_test_all.append(d['y_test'])
        print(f"  ✅ {name}: Train={d['X_train'].shape[0]}, Val={d['X_val'].shape[0]}, Test={d['X_test'].shape[0]}")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_val = np.concatenate(X_val_all, axis=0)
    y_val = np.concatenate(y_val_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)
    print(f"\n📊 Combined: Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=64, shuffle=False)

    # 2. Train centralized model
    model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 50
    best_val_f1 = 0
    best_state = None

    print(f"\n🏋️ Training centralized model for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(DEVICE))
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.numpy())

        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        val_weighted_f1 = f1_score(val_labels, val_preds, average='weighted')

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/total:.4f} | "
                  f"Train Acc: {correct/total:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Weighted F1: {val_weighted_f1:.4f}")

    # 3. Evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(DEVICE))
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_labels.extend(y.numpy())

    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    weighted_f1 = f1_score(test_labels, test_preds, average='weighted')

    print(f"\n{'='*80}")
    print(f"🏆 CENTRALIZED LEARNING RESULTS")
    print(f"{'='*80}")
    print(f"  Test Macro F1:    {macro_f1:.4f}")
    print(f"  Test Weighted F1: {weighted_f1:.4f}")
    print(f"{'='*80}")

    results = {"Metric": ["Macro F1", "Weighted F1"], "Score": [macro_f1, weighted_f1]}
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "centralized_results.csv"), index=False)
    print(f"📁 Results saved to {RESULTS_DIR}/centralized_results.csv")
