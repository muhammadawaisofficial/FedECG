"""
Table 2 Master Script: 6-Fold Out-of-Distribution (Leave-One-Hospital-Out) Cross-Validation
Framework: Proposed FedBN + 2-Layer LSTM + Attention Backbone
Evaluates Generalization on completely unseen hospital domains across all 6 folds.
"""

import os, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

BALANCED_DIR = os.path.abspath("./data/balanced")

DATA_FILES = {
    "Chapman": "chapman_4_classes_balanced.npz",
    "CPSC": "cpsc_clean.npz",
    "Georgia": "georgia_clean.npz",
    "Ningbo": "ningbo_clean.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

# 1. Model Architecture
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
        self.block = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels), ConvBlock(channels, channels))
    def forward(self, x): return x + self.block(x)

class CNN1DAttentionEnhancedLSTM(nn.Module):
    def __init__(self, input_length=3100, num_classes=3):
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
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

BATCH_SIZE = 128
def train_tensor(model, X, y, criterion, optimizer, epochs=3):
    model.train()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            bx, by = X[idx].to(DEVICE), y[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

def eval_tensor(model, X, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            bx = X[i:i+BATCH_SIZE].to(DEVICE)
            out = model(bx)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
    return y.numpy(), np.array(preds)

def run_single_ood_fold(unseen_target, client_tensors):
    train_clients = [c for c in DATA_FILES.keys() if c != unseen_target]
    print("\n" + "=" * 70)
    print(f"🚀 Training FedBN on 5 Hospitals: {train_clients}")
    print(f"🎯 Target Unseen Domain: [{unseen_target}]")
    print("=" * 70)

    global_model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
    client_models = {c: copy.deepcopy(global_model) for c in train_clients}

    bn_keys = [k for k in global_model.state_dict().keys() if 'bn' in k or 'num_batches_tracked' in k]
    non_bn_keys = [k for k in global_model.state_dict().keys() if k not in bn_keys]

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_non_bn_weights = None
    start_t = time.time()

    for r in range(20):
        val_accs = []
        for c in train_clients:
            opt = torch.optim.Adam(client_models[c].parameters(), lr=1e-3)
            train_tensor(client_models[c], client_tensors[c]["X_train"], client_tensors[c]["y_train"], criterion, opt, epochs=3)
            y_t, y_p = eval_tensor(client_models[c], client_tensors[c]["X_val"], client_tensors[c]["y_val"])
            val_accs.append(accuracy_score(y_t, y_p))

        # FedBN Aggregation
        avg_non_bn = {}
        for k in non_bn_keys:
            avg_non_bn[k] = torch.stack([client_models[c].state_dict()[k].float() for c in train_clients]).mean(0)

        for c in train_clients:
            st = client_models[c].state_dict()
            for k in non_bn_keys: st[k] = avg_non_bn[k]
            client_models[c].load_state_dict(st)

        mean_val = np.mean(val_accs)
        if mean_val > best_val_acc:
            best_val_acc = mean_val
            best_non_bn_weights = copy.deepcopy(avg_non_bn)

        if (r + 1) % 5 == 0 or r == 0:
            print(f"  Round {r+1:02d}/20 | FedBN 5-Hospital Val Acc: {mean_val:.4f} | Best: {best_val_acc:.4f}")

    # Evaluate on Unseen Domain
    eval_model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
    st = eval_model.state_dict()
    for k in non_bn_keys: st[k] = best_non_bn_weights[k]
    eval_model.load_state_dict(st)

    # Estimate BN stats on unseen domain
    eval_model.train()
    with torch.no_grad():
        for i in range(0, min(len(client_tensors[unseen_target]["X_val"]), 512), 128):
            bx = client_tensors[unseen_target]["X_val"][i:i+128].to(DEVICE)
            _ = eval_model(bx)

    y_t, y_p = eval_tensor(eval_model, client_tensors[unseen_target]["X_test"], client_tensors[unseen_target]["y_test"])

    w_f1 = f1_score(y_t, y_p, average='weighted')
    m_f1 = f1_score(y_t, y_p, average='macro')
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, average='weighted', zero_division=0)
    rec = recall_score(y_t, y_p, average='weighted', zero_division=0)

    elapsed = (time.time() - start_t) / 60
    print(f"\n🏆 Results for Unseen Target [{unseen_target}] (Time: {elapsed:.1f} min):")
    print(f"   Weighted F1: {w_f1:.4f} | Macro F1: {m_f1:.4f} | Acc: {acc:.4f}")
    print(f"   Precision:   {prec:.4f} | Recall:   {rec:.4f}")

    return {
        "Unseen Hospital": unseen_target,
        "Weighted F1": round(w_f1, 4),
        "Macro F1": round(m_f1, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4)
    }

def main():
    print("=" * 80)
    print("🌍 TABLE 2: OUT-OF-DISTRIBUTION (LEAVE-ONE-OUT) 6-FOLD BENCHMARK")
    print("=" * 80)

    client_tensors = {}
    for name, filename in DATA_FILES.items():
        p = os.path.join(BALANCED_DIR, filename)
        d = np.load(p)
        client_tensors[name] = {
            "X_train": torch.tensor(d['X_train'], dtype=torch.float32),
            "y_train": torch.tensor(d['y_train'], dtype=torch.long),
            "X_val": torch.tensor(d['X_val'], dtype=torch.float32),
            "y_val": torch.tensor(d['y_val'], dtype=torch.long),
            "X_test": torch.tensor(d['X_test'], dtype=torch.float32),
            "y_test": torch.tensor(d['y_test'], dtype=torch.long)
        }

    all_hospitals = ["CPSC", "Chapman", "Georgia", "Ningbo", "PhysioNet2017", "PTB"]
    results = []
    total_start = time.time()

    for h in all_hospitals:
        res = run_single_ood_fold(h, client_tensors)
        results.append(res)

    df = pd.DataFrame(results)
    mean_row = {
        "Unseen Hospital": "OVERALL MEAN (FedBN OOD)",
        "Weighted F1": round(df["Weighted F1"].mean(), 4),
        "Macro F1": round(df["Macro F1"].mean(), 4),
        "Accuracy": round(df["Accuracy"].mean(), 4),
        "Precision": round(df["Precision"].mean(), 4),
        "Recall": round(df["Recall"].mean(), 4)
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    print("\n" + "=" * 80)
    print(f"🎉 FINAL 6-FOLD FedBN OUT-OF-DISTRIBUTION TABLE (Total Time: {(time.time()-total_start)/60:.1f} min)")
    print("=" * 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
