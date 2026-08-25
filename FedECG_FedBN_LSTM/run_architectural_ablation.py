import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, copy
import pandas as pd
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.abspath("./data/balanced")
RESULTS_DIR = os.path.abspath("./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_DATA_PATHS = {
    "Chapman": os.path.join(DATA_DIR, "chapman_4_classes_balanced.npz"),
    "CPSC": os.path.join(DATA_DIR, "cpsc_clean.npz"),
    "Georgia": os.path.join(DATA_DIR, "georgia_clean.npz"),
    "Ningbo": os.path.join(DATA_DIR, "ningbo_clean.npz"),
    "PhysioNet2017": os.path.join(DATA_DIR, "preprocessed_physionet2017_3class.npz"),
    "PTB": os.path.join(DATA_DIR, "ptb_clean.npz")
}

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx): return torch.tensor(self.signals[idx]), torch.tensor(self.labels[idx])

# Common modules
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1), weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
    def forward(self, x): return self.dropout(F.leaky_relu(self.bn(self.conv(x))).unsqueeze(2)).squeeze(2)

class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.block = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels), ConvBlock(channels, channels))
    def forward(self, x): return x + self.block(x)

# ---------------------------------------------------------
# ABLATION ARCHITECTURE (Table 11: GRU, hidden=64, no recurrent dropout)
# ---------------------------------------------------------
class AblationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(AblationModel, self).__init__()
        self.block1 = nn.Sequential(ConvBlock(12, 16), ResidualUnit(16), nn.MaxPool1d(kernel_size=2))
        self.block2 = nn.Sequential(ConvBlock(16, 32), ResidualUnit(32), nn.MaxPool1d(kernel_size=2))
        self.block3 = nn.Sequential(ConvBlock(32, 64), ResidualUnit(64), nn.MaxPool1d(kernel_size=2))
        self.block4 = nn.Sequential(ConvBlock(64, 128, kernel_size=7), ResidualUnit(128), nn.MaxPool1d(kernel_size=2))
        # GRU instead of LSTM, hidden_size=64 instead of 32, dropout=0
        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False, dropout=0)
        self.attention = Attention(input_dim=64)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x)))).permute(0, 2, 1)
        x, _ = self.gru(x)
        x, _ = self.attention(x)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

def train_centralized(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total += y.size(0)
        val_acc = va_correct / va_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    return model

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 RUNNING: TABLE 11 ARCHITECTURAL ABLATION")
    print("="*80)

    # Combine ALL datasets for centralized training
    X_tr_all, y_tr_all, X_va_all, y_val_all = [], [], [], []
    for path in ALL_DATA_PATHS.values():
        d = np.load(path)
        X_tr_all.append(d['X_train']); y_tr_all.append(d['y_train'])
        X_va_all.append(d['X_val']); y_val_all.append(d['y_val'])

    full_train_loader = DataLoader(ECGDataset(np.concatenate(X_tr_all, axis=0), np.concatenate(y_tr_all, axis=0)), batch_size=64, shuffle=True)
    full_val_loader = DataLoader(ECGDataset(np.concatenate(X_va_all, axis=0), np.concatenate(y_val_all, axis=0)), batch_size=64, shuffle=False)

    ablation_model = AblationModel().to(DEVICE)
    print("Training Ablation Model (GRU, hidden=64, no recurrent dropout)...")
    ablation_model = train_centralized(ablation_model, full_train_loader, full_val_loader, epochs=50)

    t11_results = []
    ablation_model.eval()
    for client, path in ALL_DATA_PATHS.items():
        d = np.load(path)
        c_test_loader = DataLoader(ECGDataset(d['X_test'], d['y_test']), batch_size=128, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for x, y in c_test_loader:
                preds.extend(ablation_model(x.to(DEVICE)).argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        client_wf1 = float(f1_score(labels, preds, average='weighted'))
        t11_results.append({"Client": client, "Weighted F1": client_wf1})
        print(f"[{client}] Ablation Weighted F1: {client_wf1:.4f}")

    df_t11 = pd.DataFrame(t11_results)
    df_t11.to_csv(os.path.join(RESULTS_DIR, "table11_architectural_ablation.csv"), index=False)
    print("\n✅ TABLE 11 SAVED TO RESULTS FOLDER!")
