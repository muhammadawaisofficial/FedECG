"""
Table 1 Master Script: In-Distribution Multi-Hospital Collaborative Training
Framework: Proposed FedBN (Private Client BatchNorm + Shared 2-Layer LSTM + Attention)
Dataset: Balanced Multi-Hospital Benchmark (50 Global Rounds)
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
SAVED_MODELS_DIR = os.path.abspath("./saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

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
def train_tensor(model, X, y, criterion, optimizer, epochs=5):
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

def main():
    print("=" * 80)
    print("🌍 TABLE 1: IN-DISTRIBUTION 6-HOSPITAL FedBN + LSTM COLLABORATIVE TRAINING")
    print("=" * 80)

    # 1. Load Data into RAM
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

    all_clients = list(DATA_FILES.keys())
    global_model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
    client_models = {c: copy.deepcopy(global_model) for c in all_clients}

    bn_keys = [k for k in global_model.state_dict().keys() if 'bn' in k or 'num_batches_tracked' in k]
    non_bn_keys = [k for k in global_model.state_dict().keys() if k not in bn_keys]

    print(f"📊 Participating Hospitals: {all_clients}")
    print(f"🔒 Isolated Client BN Keys: {len(bn_keys)} | 🌐 Shared Non-BN Keys: {len(non_bn_keys)}")

    criterion = nn.CrossEntropyLoss()
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 5
    LEARNING_RATE = 1e-3
    best_val_acc = 0.0
    best_non_bn_weights = None
    best_client_bn_states = None

    start_time = time.time()

    for r in range(NUM_ROUNDS):
        val_accs = []
        for c in all_clients:
            opt = torch.optim.Adam(client_models[c].parameters(), lr=LEARNING_RATE)
            train_tensor(client_models[c], client_tensors[c]["X_train"], client_tensors[c]["y_train"], criterion, opt, epochs=LOCAL_EPOCHS)
            y_t, y_p = eval_tensor(client_models[c], client_tensors[c]["X_val"], client_tensors[c]["y_val"])
            val_accs.append(accuracy_score(y_t, y_p))

        # FedBN Aggregation: Server averages ONLY non-BN weights
        avg_non_bn = {}
        for k in non_bn_keys:
            avg_non_bn[k] = torch.stack([client_models[c].state_dict()[k].float() for c in all_clients]).mean(0)

        # Distribute shared weights while preserving private client BNs
        for c in all_clients:
            st = client_models[c].state_dict()
            for k in non_bn_keys: st[k] = avg_non_bn[k]
            client_models[c].load_state_dict(st)

        mean_val = np.mean(val_accs)
        if mean_val > best_val_acc:
            best_val_acc = mean_val
            best_non_bn_weights = copy.deepcopy(avg_non_bn)
            best_client_bn_states = {c: {k: client_models[c].state_dict()[k].clone() for k in bn_keys} for c in all_clients}

        if (r + 1) % 5 == 0 or r == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"  Round {r+1:02d}/{NUM_ROUNDS} | Avg Val Acc: {mean_val:.4f} | Peak Best: {best_val_acc:.4f} | Elapsed: {elapsed:.1f} min")

    # Final Test Evaluation
    print("\n" + "=" * 80)
    print("🏆 FINAL IN-DISTRIBUTION TEST EVALUATION (TABLE 1)")
    print("=" * 80)

    final_results = []
    for c in all_clients:
        test_model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
        st = test_model.state_dict()
        for k in non_bn_keys: st[k] = best_non_bn_weights[k]
        for k in bn_keys: st[k] = best_client_bn_states[c][k]
        test_model.load_state_dict(st)

        y_t, y_p = eval_tensor(test_model, client_tensors[c]["X_test"], client_tensors[c]["y_test"])
        final_results.append({
            "Hospital Site": c,
            "Weighted F1": round(f1_score(y_t, y_p, average='weighted'), 4),
            "Macro F1": round(f1_score(y_t, y_p, average='macro'), 4),
            "Accuracy": round(accuracy_score(y_t, y_p), 4),
            "Precision": round(precision_score(y_t, y_p, average='weighted', zero_division=0), 4),
            "Recall": round(recall_score(y_t, y_p, average='weighted', zero_division=0), 4)
        })

    df = pd.DataFrame(final_results)
    mean_row = {
        "Hospital Site": "OVERALL MEAN (FedBN Proposed)",
        "Weighted F1": round(df["Weighted F1"].mean(), 4),
        "Macro F1": round(df["Macro F1"].mean(), 4),
        "Accuracy": round(df["Accuracy"].mean(), 4),
        "Precision": round(df["Precision"].mean(), 4),
        "Recall": round(df["Recall"].mean(), 4)
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save Best Model Checkpoint
    save_path = os.path.join(SAVED_MODELS_DIR, "best_fedbn_model.pth")
    torch.save({
        "non_bn_state": best_non_bn_weights,
        "client_bn_states": best_client_bn_states,
        "best_val_acc": best_val_acc
    }, save_path)

    print(df.to_string(index=False))
    print(f"\n💾 Model Checkpoint Saved -> {save_path}")

if __name__ == "__main__":
    main()
