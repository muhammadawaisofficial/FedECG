"""
Real-World Clinical Benchmark: 50-Round FedBN + LSTM on UNBALANCED Dataset
Evaluates Federated Learning under severe clinical class imbalance across 45,000+ patient ECG records.
"""

import os, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

UNBALANCED_DIR = os.path.abspath("./data/unbalanced")
SAVED_MODELS_DIR = os.path.abspath("./saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

UNBALANCED_FILES = {
    "Chapman": "chapman_preprocessed_4_labels.npz",
    "CPSC": "cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": "georgia_preprocessed_4_labels.npz",
    "Ningbo": "ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

# Model Architecture (2-Layer LSTM Backbone)
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

def evaluate_loader(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE)
            outputs = model(bx)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(by.numpy())
    return np.array(all_labels), np.array(all_preds)

def main():
    print("=" * 80)
    print("🌍 UNBALANCED REAL-WORLD CLINICAL MULTI-HOSPITAL FedBN TRAINING (50 Rounds)")
    print("=" * 80)

    BATCH_SIZE = 128
    client_loaders = {}
    
    print("\n⚡ Caching all Unbalanced Hospital Datasets into RAM...")
    for client, filename in UNBALANCED_FILES.items():
        p = os.path.join(UNBALANCED_DIR, filename)
        data = np.load(p)
        
        tr_ds = TensorDataset(torch.tensor(data['X_train'], dtype=torch.float32), torch.tensor(data['y_train'], dtype=torch.long))
        va_ds = TensorDataset(torch.tensor(data['X_val'], dtype=torch.float32), torch.tensor(data['y_val'], dtype=torch.long))
        te_ds = TensorDataset(torch.tensor(data['X_test'], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long))
        
        client_loaders[client] = {
            "train": DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True),
            "val": DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True),
            "test": DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        }
        print(f"  🏥 {client:15s} | Train: {len(tr_ds):6d} | Val: {len(va_ds):5d} | Test: {len(te_ds):5d}")

    all_clients = list(UNBALANCED_FILES.keys())
    global_model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
    client_models = {c: copy.deepcopy(global_model) for c in all_clients}

    bn_keys = [k for k in global_model.state_dict().keys() if 'bn' in k or 'num_batches_tracked' in k]
    non_bn_keys = [k for k in global_model.state_dict().keys() if k not in bn_keys]

    criterion = nn.CrossEntropyLoss()
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 3
    LEARNING_RATE = 1e-3
    best_val_acc = 0.0
    best_non_bn_weights = None
    best_client_bn_states = None

    start_time = time.time()

    for r in range(NUM_ROUNDS):
        val_accs = []
        for c in all_clients:
            opt = torch.optim.Adam(client_models[c].parameters(), lr=LEARNING_RATE)
            client_models[c].train()
            for _ in range(LOCAL_EPOCHS):
                for bx, by in client_loaders[c]["train"]:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(client_models[c](bx), by)
                    loss.backward()
                    opt.step()
                    
            y_t, y_p = evaluate_loader(client_models[c], client_loaders[c]["val"])
            val_accs.append(accuracy_score(y_t, y_p))

        # FedBN Aggregation
        avg_non_bn = {}
        for k in non_bn_keys:
            avg_non_bn[k] = torch.stack([client_models[c].state_dict()[k].float() for c in all_clients]).mean(0)

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
            print(f"  Round {r+1:02d}/{NUM_ROUNDS} | Unbalanced Val Acc: {mean_val:.4f} | Peak Best: {best_val_acc:.4f} | Elapsed: {elapsed:.1f} min")

    # Test Evaluation
    print("\n" + "=" * 80)
    print("🏆 FINAL UNBALANCED DATASET TEST EVALUATION")
    print("=" * 80)

    final_results = []
    for c in all_clients:
        test_model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
        st = test_model.state_dict()
        for k in non_bn_keys: st[k] = best_non_bn_weights[k]
        for k in bn_keys: st[k] = best_client_bn_states[c][k]
        test_model.load_state_dict(st)

        y_t, y_p = evaluate_loader(test_model, client_loaders[c]["test"])
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
        "Hospital Site": "OVERALL MEAN (UNBALANCED FedBN)",
        "Weighted F1": round(df["Weighted F1"].mean(), 4),
        "Macro F1": round(df["Macro F1"].mean(), 4),
        "Accuracy": round(df["Accuracy"].mean(), 4),
        "Precision": round(df["Precision"].mean(), 4),
        "Recall": round(df["Recall"].mean(), 4)
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    save_path = os.path.join(SAVED_MODELS_DIR, "best_fedbn_unbalanced_model.pth")
    torch.save({
        "non_bn_state": best_non_bn_weights,
        "client_bn_states": best_client_bn_states,
        "best_val_acc": best_val_acc
    }, save_path)

    print(df.to_string(index=False))
    print(f"\n💾 Unbalanced Checkpoint Saved -> {save_path}")

if __name__ == "__main__":
    main()
