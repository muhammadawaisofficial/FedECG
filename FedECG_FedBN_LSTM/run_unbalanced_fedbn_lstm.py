import torch
import torch.nn as nn
import copy, os, time, json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*80)
print(f"🏆 UNBALANCED FedBN + LSTM (TABLE 5 & 6)")
print(f"🚀 Device: {DEVICE}")
print("="*80)

DATA_DIR = os.path.abspath("./data/unbalanced")

UNBALANCED_FILES = {
    "Chapman": "chapman_preprocessed_4_labels.npz",
    "CPSC": "cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": "georgia_preprocessed_4_labels.npz",
    "Ningbo": "ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

ALL_DATA = {k: os.path.join(DATA_DIR, v) for k, v in UNBALANCED_FILES.items()}

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx): return torch.tensor(self.signals[idx]), torch.tensor(self.labels[idx])

def load_npz_dataset(npz_path):
    data = np.load(npz_path)
    return ECGDataset(data['X_train'], data['y_train']), ECGDataset(data['X_val'], data['y_val']), ECGDataset(data['X_test'], data['y_test'])

def get_client_dataloaders(dataset_paths, batch_size):
    client_loaders = {}
    for client_name, path in dataset_paths.items():
        train_set, val_set, test_set = load_npz_dataset(path)
        client_loaders[client_name] = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
        }
    return client_loaders

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

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            total_loss += criterion(outputs, y).item() * y.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def is_bn_key(key): return ".bn." in key

def average_non_bn_weights(local_non_bn_weights):
    avg_weights = copy.deepcopy(local_non_bn_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_non_bn_weights)): avg_weights[key] += local_non_bn_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_non_bn_weights))
    return avg_weights

SAVED_MODELS_DIR = os.path.abspath("./saved_models/unbalanced")
RESULTS_DIR = os.path.abspath("./results/unbalanced")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE, LOCAL_EPOCHS, NUM_ROUNDS, LEARNING_RATE = 64, 5, 50, 1e-3

client_loaders_all = get_client_dataloaders(ALL_DATA, BATCH_SIZE)
client_names_all = list(client_loaders_all.keys())
criterion = nn.CrossEntropyLoss()

print("\n" + "="*80)
print("🌍 UNBALANCED FEDERATED IN-DISTRIBUTION (TABLE 5 & 6)")
print("="*80)

template_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
full_state = template_model.state_dict()
bn_keys = [k for k in full_state.keys() if is_bn_key(k)]
non_bn_keys = [k for k in full_state.keys() if not is_bn_key(k)]

client_bn_states_t1 = {name: {k: full_state[k].clone() for k in bn_keys} for name in client_names_all}
global_non_bn_state_t1 = {k: full_state[k].clone() for k in non_bn_keys}
best_val_acc_t1, best_non_bn_t1, best_client_bns_t1, start_round_t1 = 0, None, None, 0
checkpoint_path_t1 = os.path.join(SAVED_MODELS_DIR, "unbalanced_latest_checkpoint.pth")

if os.path.exists(checkpoint_path_t1):
    ckpt = torch.load(checkpoint_path_t1, map_location=DEVICE, weights_only=False)
    start_round_t1, best_val_acc_t1 = ckpt['round'] + 1, ckpt['best_val_acc']
    best_non_bn_t1, best_client_bns_t1 = ckpt['best_non_bn'], ckpt['best_client_bns']
    global_non_bn_state_t1, client_bn_states_t1 = ckpt['global_non_bn_state'], ckpt['client_bn_states']

start_time_t1 = time.time()
for r in range(start_round_t1, NUM_ROUNDS):
    local_non_bn_weights, local_val_accs = [], []
    for client_name in client_names_all:
        loaders = client_loaders_all[client_name]
        local_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
        local_state = local_model.state_dict()
        for k in non_bn_keys: local_state[k] = global_non_bn_state_t1[k].clone()
        for k in bn_keys: local_state[k] = client_bn_states_t1[client_name][k].clone()
        local_model.load_state_dict(local_state)

        optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
        for _ in range(LOCAL_EPOCHS): train_one_epoch(local_model, loaders["train"], criterion, optimizer)
        _, val_acc = evaluate(local_model, loaders["val"], criterion)
        local_val_accs.append(val_acc)

        updated_state = local_model.state_dict()
        client_bn_states_t1[client_name] = {k: updated_state[k].clone() for k in bn_keys}
        local_non_bn_weights.append({k: updated_state[k].clone() for k in non_bn_keys})

    global_non_bn_state_t1 = average_non_bn_weights(local_non_bn_weights)
    avg_val_acc = np.mean(local_val_accs)
    if avg_val_acc > best_val_acc_t1:
        best_val_acc_t1 = avg_val_acc
        best_non_bn_t1 = copy.deepcopy(global_non_bn_state_t1)
        best_client_bns_t1 = copy.deepcopy(client_bn_states_t1)

    elapsed = (time.time() - start_time_t1) / 60
    print(f"  🔄 [Unbalanced] Round {r+1:02d}/50 | Avg Val: {avg_val_acc:.4f} | Best: {best_val_acc_t1:.4f} | Time: {elapsed:.1f} min")

    torch.save({
        "round": r, "best_val_acc": best_val_acc_t1, "best_non_bn": best_non_bn_t1, "best_client_bns": best_client_bns_t1,
        "global_non_bn_state": global_non_bn_state_t1, "client_bn_states": client_bn_states_t1
    }, checkpoint_path_t1)

final_results_t1 = []
all_preds_global, all_labels_global = [], []
for client_name in client_names_all:
    test_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
    st = test_model.state_dict()
    for k in non_bn_keys: st[k] = best_non_bn_t1[k]
    for k in bn_keys: st[k] = best_client_bns_t1[client_name][k]
    test_model.load_state_dict(st)

    test_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in client_loaders_all[client_name]["test"]:
            outputs = test_model(x.to(DEVICE))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.to(DEVICE).cpu().numpy())

    all_preds_global.extend(all_preds)
    all_labels_global.extend(all_labels)

    final_results_t1.append({
        "Hospital Site": client_name,
        "Macro F1": float(f1_score(all_labels, all_preds, average='macro'))
    })

df_t1 = pd.DataFrame(final_results_t1)
df_t1.to_csv(os.path.join(RESULTS_DIR, "table6_unbalanced_macro.csv"), index=False)

print("\n" + "="*80)
print(f"🎉 TABLE 5 (UNBALANCED WEIGHTED F1): {f1_score(all_labels_global, all_preds_global, average='weighted'):.4f}")
print("="*80)
