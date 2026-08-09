# evaluate_global_3class.py
# =========================
# IMPORTS
# =========================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import warnings

# =========================
# CONFIG (UPDATED PATHS)
# =========================
MODEL_PATH = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity_balanced\saved_models\best_federated_model.pth"
OUTPUT_DIR = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity_balanced\GLOBAL ROC and PRECISION"

DATA_PATHS = {
    "Chapman": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_chapman_shaoxing_3_classes_balanced\pre_processed_data\chapman_4_classes_balanced.npz",
    "CPSC": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_CPSC_3_classes_balanced\pre_processed_data\cpsc_clean.npz",
    "Georgia": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_Georgia_3_classes_balanced\pre_processed_data\georgia_clean.npz",
    "Ningbo": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_Ningbo_3_classes_balanced\pre_processed_data\ningbo_clean.npz",
    "PhysioNet2017": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_Physionet_2017_4_classes\pre_processed_data\preprocessed_physionet2017_3class.npz",
    "PTB": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_PTB_3_classes_balanced\pre_processed_data\ptb_clean.npz"
}

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
CLASS_NAMES = {0: "Normal", 1: "Atrial Fibrillation", 2: "Other Rhythm"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# =========================
# MODEL DEFINITION (reduced)
# =========================
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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
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
    def __init__(self, input_length, num_classes=NUM_CLASSES):
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

        self.gru = nn.GRU(input_size=128, hidden_size=32, num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)
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

# =========================
# LOAD MODEL
# =========================
def load_model(path, input_length):
    model = CNN1DAttentionEnhanced(input_length=input_length, num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

# =========================
# LOAD TEST DATA
# =========================
def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    if "X_test" in data and "y_test" in data:
        x_test, y_test = data["X_test"], data["y_test"]
    elif "x_test" in data and "y_test" in data:
        x_test, y_test = data["x_test"], data["y_test"]
    elif "X" in data and "y" in data:
        x_test, y_test = data["X"], data["y"]
    else:
        raise KeyError(f"No test set found in {path}. Keys available: {list(data.keys())}")

    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)
    if x_test_t.ndim == 3:
        input_length = x_test_t.shape[2]
    else:
        raise ValueError(f"Unexpected x_test shape {x_test_t.shape} in {path}. Expected (N,12,L).")
    return test_loader, input_length

# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model, test_loader):
    y_true, y_score = [], []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            probs = softmax(outputs)
            y_true.extend(y.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    return y_true, y_score, y_true_bin

# =========================
# PLOT ROC (high clarity)
# =========================
def plot_roc(y_true_bin, y_score, dataset_name, save_dir):
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 6), dpi=150)
    plotted_any = False

    for i in range(NUM_CLASSES):
        if np.sum(y_true_bin[:, i]) == 0:
            print(f"Warning: No positive samples for class {i} in {dataset_name}.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=3,
                 label=f"{CLASS_NAMES.get(i, f'Class {i}')} (AUC={roc_auc[i]:.2f})")

        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", lw=2)

    # ❌ Removed grid to avoid grey/black background
    # plt.grid(True, linestyle="--", alpha=0.5)

    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)

    plt.legend(
        loc="lower right",
        fontsize=15,
        frameon=False,
        borderpad=0,
        handlelength=2,
    )

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_roc.png"),
                bbox_inches="tight", dpi=600)
    plt.close()



# =========================
# PLOT PRECISION-RECALL (high clarity)
# =========================
def plot_pr(y_true_bin, y_score, dataset_name, save_dir):
    precision, recall, avg_precision = {}, {}, {}
    plt.figure(figsize=(8, 6), dpi=150)
    plotted_any = False

    for i in range(NUM_CLASSES):
        if np.sum(y_true_bin[:, i]) == 0:
            continue
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=3,
                 label=f"{CLASS_NAMES.get(i, f'Class {i}')} (AP={avg_precision[i]:.2f})")
        plotted_any = True

    if np.sum(y_true_bin) > 0:
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_score.ravel())
        avg_precision["micro"] = average_precision_score(
            y_true_bin, y_score, average="micro")
        plt.plot(recall["micro"], precision["micro"], linestyle="--", lw=3,
                 label=f"Micro-average (AP={avg_precision['micro']:.2f})")
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    # ❌ Removed grid to avoid grey/black boxes
    # plt.grid(True, linestyle="--", alpha=0.5)

    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)

    plt.legend(
        loc="best",
        fontsize=15,
        frameon=False,
        borderpad=0,
        handlelength=2,
    )

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_pr.png"),
                bbox_inches="tight", dpi=600)
    plt.close()



# =========================
# MAIN
# =========================
if __name__ == "__main__":
    all_y_true, all_y_score, all_y_bin = [], [], []

    for dataset_name, path in DATA_PATHS.items():
        print(f"\nEvaluating {dataset_name} (using {NUM_CLASSES}-class model)...")
        if not os.path.exists(path):
            print(f"Dataset file not found: {path}")
            continue

        test_loader, input_length = load_dataset(path)
        model = load_model(MODEL_PATH, input_length)
        y_true, y_score, y_true_bin = evaluate_model(model, test_loader)

        plot_roc(y_true_bin, y_score, dataset_name, OUTPUT_DIR)
        plot_pr(y_true_bin, y_score, dataset_name, OUTPUT_DIR)

        try:
            auc_score = roc_auc_score(y_true_bin, y_score, average="macro")
            print(f"{dataset_name} Macro-AUC: {auc_score:.4f}")
        except:
            pass

        if y_true.size > 0:
            all_y_true.append(y_true)
            all_y_score.append(y_score)
            all_y_bin.append(y_true_bin)

    if len(all_y_true) > 0:
        print("\nPerforming GLOBAL evaluation across all datasets...")
        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_score = np.concatenate(all_y_score, axis=0)
        all_y_bin = np.concatenate(all_y_bin, axis=0)

        plot_roc(all_y_bin, all_y_score, "GLOBAL", OUTPUT_DIR)
        plot_pr(all_y_bin, all_y_score, "GLOBAL", OUTPUT_DIR)

        try:
            global_auc = roc_auc_score(all_y_bin, all_y_score, average="macro")
            print(f"GLOBAL Macro-AUC: {global_auc:.4f}")
        except:
            pass

    print("\nDone. Plots saved to:", OUTPUT_DIR)


