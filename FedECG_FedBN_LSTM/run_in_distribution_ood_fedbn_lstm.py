import os, sys, copy, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*80)
print(f"🚀 FEDECG MASTER 50-ROUND SUITE (BLACKWELL GPU ACCELERATED)")
print(f"🚀 Hardware: {DEVICE} ({torch.cuda.get_device_name(0)}) | Batch Size: 64 | Local Epochs: 5 | Rounds: 50")
print("="*80)

DATA_DIR = os.path.abspath("./data/balanced")
SAVED_MODELS_DIR = os.path.abspath("./saved_models")
RESULTS_DIR = os.path.abspath("./results")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "ood_fold_metrics"), exist_ok=True)

DATA_FILES = {
    "Chapman": "chapman_4_classes_balanced.npz",
    "CPSC": "cpsc_clean.npz",
    "Georgia": "georgia_clean.npz",
    "Ningbo": "ningbo_clean.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

# --- 1. Model Architecture ---
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

# --- 2. Training Helpers ---
BATCH_SIZE = 64   # Exact match to paper for maximum accuracy
LOCAL_EPOCHS = 5  # Exact match to paper
NUM_ROUNDS = 50   # Full 50 communication rounds

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

print("\n⏳ Fast-loading all datasets directly into memory...")
client_tensors = {}
for name, filename in DATA_FILES.items():
    d = np.load(os.path.join(DATA_DIR, filename))
    client_tensors[name] = {
        "X_train": torch.tensor(d['X_train'], dtype=torch.float32), "y_train": torch.tensor(d['y_train'], dtype=torch.long),
        "X_val": torch.tensor(d['X_val'], dtype=torch.float32), "y_val": torch.tensor(d['y_val'], dtype=torch.long),
        "X_test": torch.tensor(d['X_test'], dtype=torch.float32), "y_test": torch.tensor(d['y_test'], dtype=torch.long)
    }
all_clients = list(DATA_FILES.keys())

dummy_model = CNN1DAttentionEnhancedLSTM()
all_keys = list(dummy_model.state_dict().keys())
bn_keys = [k for k in all_keys if 'bn' in k or 'num_batches_tracked' in k]
non_bn_keys = [k for k in all_keys if k not in bn_keys]
criterion = nn.CrossEntropyLoss()

# ==============================================================================
# 🌟 STEP 1: TABLE 2 OUT-OF-DISTRIBUTION (6-FOLD BENCHMARK - 50 ROUNDS)
# ==============================================================================
print("\n" + "="*80)
print("🌍 STEP 1: STARTING TABLE 2 (OUT-OF-DISTRIBUTION 6-FOLD BENCHMARK - 50 ROUNDS)")
print("="*80)

t2_results = []
for unseen in all_clients:
    json_path = os.path.join(RESULTS_DIR, "ood_fold_metrics", f"ood_{unseen}.json")
    fold_ckpt_path = os.path.join(SAVED_MODELS_DIR, f"ood_{unseen}_round_ckpt.pth")

    if os.path.exists(json_path):
        print(f"⏭️ Skipping Fold [{unseen}] -> Already completed and saved on disk!")
        with open(json_path, "r") as f: t2_results.append(json.load(f))
        continue

    train_clients = [c for c in all_clients if c != unseen]
    print(f"\n🚀 Fold -> Training on {train_clients} | Testing on UNSEEN [{unseen}] (50 Rounds)...")
    
    c_models = {c: CNN1DAttentionEnhancedLSTM().to(DEVICE) for c in train_clients}
    best_f_val, best_f_weights = 0.0, None
    start_round = 0

    if os.path.exists(fold_ckpt_path):
        print(f"🔄 Resuming fold from checkpoint...")
        ckpt = torch.load(fold_ckpt_path, map_location=DEVICE)
        start_round = ckpt['round'] + 1
        best_f_val = ckpt['best_f_val']
        best_f_weights = ckpt['best_f_weights']
        for c in train_clients: c_models[c].load_state_dict(ckpt['c_models_state'][c])

    start_time = time.time()
    
    for r in range(start_round, NUM_ROUNDS):
        val_accs = []
        for c in train_clients:
            opt = torch.optim.Adam(c_models[c].parameters(), lr=1e-3)
            train_tensor(c_models[c], client_tensors[c]["X_train"], client_tensors[c]["y_train"], criterion, opt, epochs=LOCAL_EPOCHS)
            y_t, y_p = eval_tensor(c_models[c], client_tensors[c]["X_val"], client_tensors[c]["y_val"])
            val_accs.append(accuracy_score(y_t, y_p))

        avg_non_bn = {k: torch.stack([c_models[c].state_dict()[k].float() for c in train_clients]).mean(0) for k in non_bn_keys}
        for c in train_clients:
            st = c_models[c].state_dict()
            for k in non_bn_keys: st[k] = avg_non_bn[k]
            c_models[c].load_state_dict(st)

        mean_val = float(np.mean(val_accs))
        if mean_val > best_f_val:
            best_f_val = mean_val
            best_f_weights = {k: torch.stack([c_models[c].state_dict()[k].float() for c in train_clients]).mean(0) for k in all_keys}
        
        elapsed = (time.time() - start_time) / 60
        print(f"  🔄 [Fold: {unseen}] Round {r+1:02d}/{NUM_ROUNDS} | Avg Val: {mean_val:.4f} | Best: {best_f_val:.4f} | Time: {elapsed:.1f} min")

        torch.save({
            "round": r, "best_f_val": best_f_val, "best_f_weights": best_f_weights,
            "c_models_state": {c: c_models[c].state_dict() for c in train_clients}
        }, fold_ckpt_path)

    print(f"  🏆 Evaluating on Unseen {unseen} with Calibrated Global BatchNorm...")
    eval_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
    eval_model.load_state_dict(best_f_weights)

    y_t, y_p = eval_tensor(eval_model, client_tensors[unseen]["X_test"], client_tensors[unseen]["y_test"])
    fold_metric = {
        "Unseen Hospital": unseen,
        "Weighted F1": round(float(f1_score(y_t, y_p, average='weighted')), 4),
        "Macro F1": round(float(f1_score(y_t, y_p, average='macro')), 4),
        "Accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "Precision": round(float(precision_score(y_t, y_p, average='weighted', zero_division=0)), 4),
        "Recall": round(float(recall_score(y_t, y_p, average='weighted', zero_division=0)), 4)
    }
    t2_results.append(fold_metric)
    
    with open(json_path, "w") as f: json.dump(fold_metric, f, indent=2)
    if os.path.exists(fold_ckpt_path): os.remove(fold_ckpt_path)
    print(f"  ✅ [{unseen}] Finished -> Weighted F1: {fold_metric['Weighted F1']} | Saved to JSON!")

df_t2 = pd.DataFrame(t2_results)
mean_row_t2 = {
    "Unseen Hospital": "OVERALL MEAN (FedBN OOD - 50 Rounds)",
    "Weighted F1": round(float(df_t2["Weighted F1"].mean()), 4),
    "Macro F1": round(float(df_t2["Macro F1"].mean()), 4),
    "Accuracy": round(float(df_t2["Accuracy"].mean()), 4),
    "Precision": round(float(df_t2["Precision"].mean()), 4),
    "Recall": round(float(df_t2["Recall"].mean()), 4)
}
df_t2 = pd.concat([df_t2, pd.DataFrame([mean_row_t2])], ignore_index=True)
df_t2.to_csv(os.path.join(RESULTS_DIR, "table2_ood_summary.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "table2_ood_summary.txt"), "w") as f: f.write(df_t2.to_string(index=False))

print("\n" + "="*80)
print("🎉🎉 STEP 1 COMPLETE! TABLE 2 (OOD) SAVED TO DISK!")
print(df_t2.to_string(index=False))
print("="*80)


# ==============================================================================
# 🌟 STEP 2: TABLE 1 IN-DISTRIBUTION TRAINING (50 ROUNDS - 5 LOCAL EPOCHS)
# ==============================================================================
print("\n" + "="*80)
print("🌍 STEP 2: AUTOMATICALLY STARTING TABLE 1 (IN-DISTRIBUTION 50-ROUND FedBN)")
print("="*80)

global_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
client_models = {c: copy.deepcopy(global_model) for c in all_clients}

start_round_t1, best_val_acc_t1 = 0, 0.0
best_non_bn_weights_t1, best_client_bn_states_t1 = None, None
checkpoint_path_t1 = os.path.join(SAVED_MODELS_DIR, "table1_latest_checkpoint.pth")

if os.path.exists(checkpoint_path_t1):
    print(f"🔄 Checkpoint found! Restoring state from {checkpoint_path_t1}...")
    ckpt = torch.load(checkpoint_path_t1, map_location=DEVICE)
    start_round_t1 = ckpt['round'] + 1
    best_val_acc_t1 = ckpt['best_val_acc']
    best_non_bn_weights_t1 = ckpt['best_non_bn_weights']
    best_client_bn_states_t1 = ckpt['best_client_bn_states']
    for c in all_clients: client_models[c].load_state_dict(ckpt['client_models_state'][c])
    print(f"✅ Successfully resumed Table 1 at Round {start_round_t1+1}.")

start_time_t1 = time.time()
for r in range(start_round_t1, 50):
    val_accs = []
    for c in all_clients:
        opt = torch.optim.Adam(client_models[c].parameters(), lr=1e-3)
        train_tensor(client_models[c], client_tensors[c]["X_train"], client_tensors[c]["y_train"], criterion, opt, epochs=LOCAL_EPOCHS)
        y_t, y_p = eval_tensor(client_models[c], client_tensors[c]["X_val"], client_tensors[c]["y_val"])
        val_accs.append(accuracy_score(y_t, y_p))

    avg_non_bn = {k: torch.stack([client_models[c].state_dict()[k].float() for c in all_clients]).mean(0) for k in non_bn_keys}
    for c in all_clients:
        st = client_models[c].state_dict()
        for k in non_bn_keys: st[k] = avg_non_bn[k]
        client_models[c].load_state_dict(st)

    mean_val = float(np.mean(val_accs))
    if mean_val > best_val_acc_t1:
        best_val_acc_t1 = mean_val
        best_non_bn_weights_t1 = copy.deepcopy(avg_non_bn)
        best_client_bn_states_t1 = {c: {k: client_models[c].state_dict()[k].clone() for k in bn_keys} for c in all_clients}

    elapsed = (time.time() - start_time_t1) / 60
    print(f"  🔄 [Table 1] Round {r+1:02d}/50 | Avg Val: {mean_val:.4f} | Peak Best: {best_val_acc_t1:.4f} | Time: {elapsed:.1f} min")

    torch.save({
        "round": r, "best_val_acc": best_val_acc_t1, "best_non_bn_weights": best_non_bn_weights_t1,
        "best_client_bn_states": best_client_bn_states_t1, "client_models_state": {c: client_models[c].state_dict() for c in all_clients}
    }, checkpoint_path_t1)

final_results_t1 = []
for c in all_clients:
    test_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
    st = test_model.state_dict()
    for k in non_bn_keys: st[k] = best_non_bn_weights_t1[k]
    for k in bn_keys: st[k] = best_client_bn_states_t1[c][k]
    test_model.load_state_dict(st)

    y_t, y_p = eval_tensor(test_model, client_tensors[c]["X_test"], client_tensors[c]["y_test"])
    final_results_t1.append({
        "Hospital Site": c,
        "Weighted F1": round(float(f1_score(y_t, y_p, average='weighted')), 4),
        "Macro F1": round(float(f1_score(y_t, y_p, average='macro')), 4),
        "Accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "Precision": round(float(precision_score(y_t, y_p, average='weighted', zero_division=0)), 4),
        "Recall": round(float(recall_score(y_t, y_p, average='weighted', zero_division=0)), 4)
    })

df_t1 = pd.DataFrame(final_results_t1)
mean_row_t1 = {
    "Hospital Site": "OVERALL MEAN (FedBN Proposed)",
    "Weighted F1": round(float(df_t1["Weighted F1"].mean()), 4),
    "Macro F1": round(float(df_t1["Macro F1"].mean()), 4),
    "Accuracy": round(float(df_t1["Accuracy"].mean()), 4),
    "Precision": round(float(df_t1["Precision"].mean()), 4),
    "Recall": round(float(df_t1["Recall"].mean()), 4)
}
df_t1 = pd.concat([df_t1, pd.DataFrame([mean_row_t1])], ignore_index=True)
df_t1.to_csv(os.path.join(RESULTS_DIR, "table1_final_results.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "table1_final_results.txt"), "w") as f: f.write(df_t1.to_string(index=False))

print("\n" + "="*80)
print("🎉🎉 FULL 50-ROUND MASTER BENCHMARK IS 100% COMPLETE!")
print("="*80)
print(df_t1.to_string(index=False))
