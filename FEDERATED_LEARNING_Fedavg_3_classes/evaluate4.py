import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

# ============================================================== #
# 🔹 New Model Definition (Reduced Complexity Version)
# ============================================================== #
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        x = self.dropout(x.unsqueeze(2)).squeeze(2)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class CNN1DAttentionEnhanced(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1DAttentionEnhanced, self).__init__()

        self.block1 = nn.Sequential(
            ConvBlock(12, 16),
            ResidualUnit(16),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            ConvBlock(16, 32),
            ResidualUnit(32),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            ConvBlock(32, 64),
            ResidualUnit(64),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block4 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=7),
            ResidualUnit(128),
            nn.MaxPool1d(kernel_size=2)
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )

        self.attention = Attention(input_dim=32)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x, _ = self.attention(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================== #
# ✅ Paths and Configurations
# ============================================================== #
DATA_PATHS = {
    "Chapman": r"D:\Engr GHANWA MS 220211001\Centralized_chapman_shaoxing_3_classes\pre_processed_data\chapman_preprocessed_4_labels.npz",
    "CPSC": r"D:\Engr GHANWA MS 220211001\Centralized_CPSC_3_classes\pre_processed_data\cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": r"D:\Engr GHANWA MS 220211001\Centralized_Georgia_3_classes\pre_processed_data\georgia_preprocessed_4_labels.npz",
    "Ningbo": r"D:\Engr GHANWA MS 220211001\Centralized_Ningbo_3_classes\pre_processed_data\ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": r"D:\Engr GHANWA MS 220211001\Centralized_Physionet_2017_4_classes\pre_processed_data\preprocessed_physionet2017_3class.npz",  
    "PTB": r"D:\Engr GHANWA MS 220211001\Centralized_PTB_3_classes\pre_processed_data\ptb_combined_preprocessed_4_labels.npz"
}

MODEL_PATH = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity\saved_models\best_federated_model.pth"
RESULTS_FOLDER = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity\uniform confusion"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Atrial Fibrillation", "Other Rhythm"]

# ============================================================== #
# ✅ Load Model
# ============================================================== #
NUM_CLASSES = 3
INPUT_LENGTH = 3100
model = CNN1DAttentionEnhanced(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ============================================================== #
# ✅ Step 1: Find smallest test set size
# ============================================================== #
test_sizes = {}
datasets_data = {}

for name, path in DATA_PATHS.items():
    data = np.load(path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    test_sizes[name] = len(X_test)
    datasets_data[name] = (X_test, y_test)

min_test_size = min(test_sizes.values())

print("\n=== Test Samples per Dataset ===")
for name, size in test_sizes.items():
    print(f"{name}: {size}")
print(f"\n✅ Using {min_test_size} samples from each dataset for uniform test set.\n")

# ============================================================== #
# ✅ Step 2: Create combined uniform test set
# ============================================================== #
X_combined, y_combined = [], []

for name, (X_test, y_test) in datasets_data.items():
    indices = np.random.choice(len(X_test), min_test_size, replace=False)
    X_combined.append(X_test[indices])
    y_combined.append(y_test[indices])

X_combined = np.concatenate(X_combined, axis=0)
y_combined = np.concatenate(y_combined, axis=0)

print(f"Combined uniform test set shape: {X_combined.shape}, Labels shape: {y_combined.shape}")

# ============================================================== #
# ✅ Step 3: Evaluate on the combined uniform test set
# ============================================================== #
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_combined, dtype=torch.float32),
                  torch.tensor(y_combined, dtype=torch.long)),
    batch_size=64, shuffle=False
)

all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ============================================================== #
# ✅ Step 4: Generate Report and Confusion Matrix
# ============================================================== #
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
cm = confusion_matrix(all_labels, all_preds)

print("\n=== Classification Report (0: Normal, 1: AFIB, 2: Other Rhythm) ===")
print(report)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Combined Uniform Test Set")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "combined_uniform_confusion_matrix.png"))
plt.show()

# Save classification report
with open(os.path.join(RESULTS_FOLDER, "combined_uniform_classification_report.txt"), "w") as f:
    f.write(report)

print(f"\n✅ Combined uniform evaluation complete! Results saved to: {RESULTS_FOLDER}")
