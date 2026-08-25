import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, copy
import pandas as pd
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*80)
print(f"🚀 FEDECG ABLATIONS & FAST TABLES (Full GPU Accelerated)")
print("="*80)

DATA_DIR = os.path.abspath("./data/balanced")
RESULTS_DIR = os.path.abspath("./results/centralized")
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

class CNN1DAttentionEnhancedLSTM(nn.Module):
    def __init__(self, input_length=3100, num_classes=3):
        super(CNN1DAttentionEnhancedLSTM, self).__init__()
        self.block1 = nn.Sequential(ConvBlock(12, 16), ResidualUnit(16), nn.MaxPool1d(kernel_size=2))
        self.block2 = nn.Sequential(ConvBlock(16, 32), ResidualUnit(32), nn.MaxPool1d(kernel_size=2))
        self.block3 = nn.Sequential(ConvBlock(32, 64), ResidualUnit(64), nn.MaxPool1d(kernel_size=2))
        self.block4 = nn.Sequential(ConvBlock(64, 128, kernel_size=7), ResidualUnit(128), nn.MaxPool1d(kernel_size=2))
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)
        self.attention = Attention(input_dim=32)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x)))).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

def train_and_evaluate(train_loader, val_loader, test_loader, lr=1e-3, epochs=50):
    model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_state, final_val_loss, final_train_acc = 0, None, 0, 0
    
    for epoch in range(epochs):
        model.train()
        tr_correct, tr_total = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += y.size(0)
        final_train_acc = tr_correct / tr_total
        
        model.eval()
        va_correct, va_total, va_loss_sum = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                va_loss_sum += criterion(out, y).item() * y.size(0)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total += y.size(0)
        val_acc = va_correct / va_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            final_val_loss = va_loss_sum / va_total
            
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.to(DEVICE).cpu().numpy())
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    return {
        "Train Accuracy": final_train_acc, "Val Accuracy": best_val_acc, "Val Loss": final_val_loss,
        "Weighted F1": float(f1_score(all_labels, all_preds, average='weighted')),
        "Macro F1": float(f1_score(all_labels, all_preds, average='macro'))
    }

if __name__ == "__main__":
    # 1. TABLE 10
    print("\n" + "="*80 + "\n📊 TABLE 10: MODEL SIZE AND PARAMETERS\n" + "="*80)
    dummy_model = CNN1DAttentionEnhancedLSTM()
    total_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    print(f"Proposed LSTM Model -> Parameters: {total_params/1e6:.2f}M | Size: {total_params * 4 / (1024 ** 2):.2f} MB")

    # 3. TABLE 3
    print("\n" + "="*80 + "\n📊 TABLE 3: SINGLE CLIENT BASELINES (Train 1, Test 1)\n" + "="*80)
    t3_results = []
    for client, path in ALL_DATA_PATHS.items():
        print(f"\nTraining isolated model for [{client}]...")
        d = np.load(path)
        res = train_and_evaluate(DataLoader(ECGDataset(d['X_train'], d['y_train']), batch_size=64, shuffle=True),
                                 DataLoader(ECGDataset(d['X_val'], d['y_val']), batch_size=64, shuffle=False),
                                 DataLoader(ECGDataset(d['X_test'], d['y_test']), batch_size=64, shuffle=False), lr=1e-3, epochs=50)
        t3_results.append({"Client": client, "Macro F1": res["Macro F1"]})
        print(f"✅ [{client}] Single Client Macro F1: {res['Macro F1']:.4f}")
    pd.DataFrame(t3_results).to_csv(os.path.join(RESULTS_DIR, "table3_single_client.csv"), index=False)

    # 4. TABLE 9
    print("\n" + "="*80 + "\n📊 TABLE 9: HYPERPARAMETER ABLATIONS\n" + "="*80)
    X_tr, y_tr, X_va, y_val, X_te, y_te = [], [], [], [], [], []
    for path in ALL_DATA_PATHS.values():
        d = np.load(path)
        X_tr.append(d['X_train']); y_tr.append(d['y_train'])
        X_va.append(d['X_val']); y_val.append(d['y_val'])
        X_te.append(d['X_test']); y_te.append(d['y_test'])

    X_train, y_train = np.concatenate(X_tr, axis=0), np.concatenate(y_tr, axis=0)
    X_val, y_val = np.concatenate(X_va, axis=0), np.concatenate(y_val, axis=0)
    X_test, y_test = np.concatenate(X_te, axis=0), np.concatenate(y_te, axis=0)

    t9_results = []
    for config in [{"lr": 0.001, "bs": 32}, {"lr": 0.002, "bs": 32}, {"lr": 0.002, "bs": 64}]:
        print(f"\nRunning Ablation -> LR: {config['lr']} | BS: {config['bs']}...")
        res = train_and_evaluate(DataLoader(ECGDataset(X_train, y_train), batch_size=config['bs'], shuffle=True),
                                 DataLoader(ECGDataset(X_val, y_val), batch_size=config['bs'], shuffle=False),
                                 DataLoader(ECGDataset(X_test, y_test), batch_size=config['bs'], shuffle=False), lr=config['lr'], epochs=50)
        t9_results.append({"LR": config['lr'], "BS": config['bs'], "Train Accuracy": res["Train Accuracy"], "Val Accuracy": res["Val Accuracy"], "Val Loss": res["Val Loss"], "Mean Test F1": res["Macro F1"]})
        print(f"✅ Result -> Train Acc: {res['Train Accuracy']:.4f} | Val Acc: {res['Val Accuracy']:.4f} | Mean Test F1: {res['Macro F1']:.4f}")
    pd.DataFrame(t9_results).to_csv(os.path.join(RESULTS_DIR, "table9_ablations.csv"), index=False)
    print("\n🎉 ALL LOCAL TABLES & ABLATIONS COMPLETE!")
