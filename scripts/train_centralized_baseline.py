"""
Centralized ("Centroid") Pooled Baseline Training Script
Establishes the theoretical upper-bound by training a single global LSTM model on all 6 pooled hospital datasets.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

# 1. Model Architecture (2-Layer LSTM + Attention Backbone)
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
    print("🧠 CENTRALIZED ('CENTROID') BASELINE POOLED TRAINING")
    print("=" * 80)
    
    client_test_loaders = {}
    pooled_x_train, pooled_y_train = [], []
    pooled_x_val, pooled_y_val = [], []
    
    BATCH_SIZE = 128
    
    for client, filename in DATA_FILES.items():
        p = os.path.join(BALANCED_DIR, filename)
        data = np.load(p)
        
        pooled_x_train.append(data['X_train'])
        pooled_y_train.append(data['y_train'])
        pooled_x_val.append(data['X_val'])
        pooled_y_val.append(data['y_val'])
        
        # Test loaders per hospital
        ts_x = torch.tensor(data['X_test'], dtype=torch.float32)
        ts_y = torch.tensor(data['y_test'], dtype=torch.long)
        client_test_loaders[client] = DataLoader(TensorDataset(ts_x, ts_y), batch_size=BATCH_SIZE, shuffle=False)

    X_train_all = torch.tensor(np.concatenate(pooled_x_train, axis=0), dtype=torch.float32)
    y_train_all = torch.tensor(np.concatenate(pooled_y_train, axis=0), dtype=torch.long)
    X_val_all = torch.tensor(np.concatenate(pooled_x_val, axis=0), dtype=torch.float32)
    y_val_all = torch.tensor(np.concatenate(pooled_y_val, axis=0), dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_all, y_train_all), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_all, y_val_all), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Pooled Training Set:   {len(X_train_all):,d} samples")
    print(f"📊 Pooled Validation Set: {len(X_val_all):,d} samples")
    
    model = CNN1DAttentionEnhancedLSTM(input_length=3100, num_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    NUM_EPOCHS = 50
    best_val_acc = 0.0
    best_weights_path = os.path.join(SAVED_MODELS_DIR, "best_centralized_model.pth")
    
    print("\n🚀 Training Centralized Model for 50 Epochs...")
    start_t = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        y_true, y_pred = evaluate_loader(model, val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{NUM_EPOCHS} | Pooled Val Acc: {val_acc:.4f} | Peak Best: {best_val_acc:.4f}")
            
    print(f"\n✅ Training complete in {(time.time()-start_t)/60:.1f} minutes!")
    
    # Load Best Model for Final Evaluation
    model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))
    
    print("\n" + "=" * 80)
    print("🏆 FINAL CENTRALIZED MODEL TEST EVALUATION")
    print("=" * 80)
    
    results = []
    for client, loader in client_test_loaders.items():
        y_t, y_p = evaluate_loader(model, loader)
        results.append({
            "Hospital Site": client,
            "Weighted F1": round(f1_score(y_t, y_p, average='weighted'), 4),
            "Macro F1": round(f1_score(y_t, y_p, average='macro'), 4),
            "Accuracy": round(accuracy_score(y_t, y_p), 4),
            "Precision": round(precision_score(y_t, y_p, average='weighted', zero_division=0), 4),
            "Recall": round(recall_score(y_t, y_p, average='weighted', zero_division=0), 4)
        })
        
    df = pd.DataFrame(results)
    mean_row = {
        "Hospital Site": "OVERALL CENTRALIZED MEAN",
        "Weighted F1": round(df["Weighted F1"].mean(), 4),
        "Macro F1": round(df["Macro F1"].mean(), 4),
        "Accuracy": round(df["Accuracy"].mean(), 4),
        "Precision": round(df["Precision"].mean(), 4),
        "Recall": round(df["Recall"].mean(), 4)
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
