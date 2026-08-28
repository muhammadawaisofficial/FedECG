import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

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

dummy_model = CNN1DAttentionEnhancedLSTM()
all_keys = list(dummy_model.state_dict().keys())
bn_keys = [k for k in all_keys if 'bn' in k or 'num_batches_tracked' in k]
non_bn_keys = [k for k in all_keys if k not in bn_keys]

# --- 2. Data Loading ---
print("Loading data...")
client_tensors = {}
for name, filename in DATA_FILES.items():
    d = np.load(os.path.join(DATA_DIR, filename))
    client_tensors[name] = {
        "X_train": torch.tensor(d['X_train'], dtype=torch.float32), "y_train": torch.tensor(d['y_train'], dtype=torch.long),
        "X_val": torch.tensor(d['X_val'], dtype=torch.float32), "y_val": torch.tensor(d['y_val'], dtype=torch.long),
        "X_test": torch.tensor(d['X_test'], dtype=torch.float32), "y_test": torch.tensor(d['y_test'], dtype=torch.long)
    }

BATCH_SIZE = 64
LOCAL_EPOCHS = 5
NUM_ROUNDS = 50

def train_fedavg(model, X, y, epochs=LOCAL_EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X[idx].to(DEVICE)), y[idx].to(DEVICE))
            loss.backward()
            optimizer.step()

def evaluate(model, X, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            out = model(X[i:i+BATCH_SIZE].to(DEVICE))
            preds.extend(torch.max(out, 1)[1].cpu().numpy())
    return preds

strategy = "FedAdam-BN"
print(f"\n{'='*50}\nSTARTING {strategy} (50 Rounds)\n{'='*50}")

global_model = CNN1DAttentionEnhancedLSTM().to(DEVICE)
client_models = {c: CNN1DAttentionEnhancedLSTM().to(DEVICE) for c in DATA_FILES.keys()}

# For FedAdam, server optimizer only updates non-BN parameters
server_optimizer = torch.optim.Adam([p for n, p in global_model.named_parameters() if n in non_bn_keys], lr=0.01)

# Initialize all clients with global non-BN weights
for c in client_models:
    st = client_models[c].state_dict()
    for k in non_bn_keys:
        st[k] = global_model.state_dict()[k]
    client_models[c].load_state_dict(st)

checkpoint_path = os.path.join(RESULTS_DIR, "fedadam_bn_checkpoint.pth")
start_round = 1

if os.path.exists(checkpoint_path):
    print("Found checkpoint! Resuming training...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    start_round = ckpt['round'] + 1
    global_model.load_state_dict(ckpt['global_model'])
    for c in client_models:
        client_models[c].load_state_dict(ckpt['client_models'][c])
    server_optimizer.load_state_dict(ckpt['server_optimizer'])
    print(f"Resuming from round {start_round}")

for round_idx in range(start_round, NUM_ROUNDS + 1):
    print(f"[{strategy}] Round {round_idx}/{NUM_ROUNDS}...")
    
    # 1. Local Training
    for client_name, data in client_tensors.items():
        model = client_models[client_name]
        
        # Sync with global non-BN before local training
        st = model.state_dict()
        for k in non_bn_keys:
            st[k] = global_model.state_dict()[k]
        model.load_state_dict(st)
        
        train_fedavg(model, data["X_train"], data["y_train"])

    # 2. Aggregation (FedAdam-BN)
    avg_non_bn = {}
    for key in non_bn_keys:
        orig_dtype = global_model.state_dict()[key].dtype
        stacked = torch.stack([client_models[c].state_dict()[key].float() for c in client_models.keys()])
        avg_non_bn[key] = stacked.mean(dim=0).to(orig_dtype)
    
    server_optimizer.zero_grad()
    for name, param in global_model.named_parameters():
        if name in non_bn_keys and param.requires_grad:
            # Pseudo-gradient: global - aggregated client
            param.grad = param.data - avg_non_bn[name]
    server_optimizer.step()
    
    # Manually update non-trainable non-bn buffers (if any exist)
    with torch.no_grad():
        for name, buffer in global_model.named_buffers():
            if name in non_bn_keys:
                buffer.copy_(avg_non_bn[name])

    # 3. Checkpointing
    torch.save({
        'round': round_idx,
        'global_model': global_model.state_dict(),
        'client_models': {c: client_models[c].state_dict() for c in client_models},
        'server_optimizer': server_optimizer.state_dict()
    }, checkpoint_path)

# 4. Final Evaluation (FedBN specific: clients use their own models on their own test sets)
print(f"Evaluating Final {strategy} Global Model...")
all_preds = []
all_trues = []

for client_name in DATA_FILES.keys():
    test_model = client_models[client_name]
    st = test_model.state_dict()
    # Sync with final global non-BN weights
    for k in non_bn_keys:
        st[k] = global_model.state_dict()[k]
    test_model.load_state_dict(st)
    
    preds = evaluate(test_model, client_tensors[client_name]["X_test"], client_tensors[client_name]["y_test"])
    all_preds.extend(preds)
    all_trues.extend(client_tensors[client_name]["y_test"].numpy())

wf1 = f1_score(all_trues, all_preds, average='weighted', zero_division=0)
print(f"{strategy} Final Weighted F1: {wf1:.4f}")

df = pd.DataFrame([{"Strategy": strategy, "Weighted_F1": wf1}])
out_csv = os.path.join(RESULTS_DIR, "fedadam_bn_result.csv")
df.to_csv(out_csv, index=False)
print(f"\nDone! Results saved to {out_csv}")
