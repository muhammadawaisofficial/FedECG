import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# =====================================
# ✅ Import model and dataset utilities
# =====================================
from torch.utils.data import DataLoader
from dataset import get_client_dataloaders
from models import CNN1DAttentionEnhanced  # Make sure this is saved in models.py or same file

# =====================================
# 🧩 Configuration
# =====================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 3
INPUT_LENGTH = 2000  # Modify to your real input length (e.g., 3100 if 10s@310Hz)
CLASS_NAMES = ["Normal", "AFIB", "Other Rhythm"]

# Paths
MODEL_PATH = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity\saved_models\best_federated_model.pth"
RESULTS_DIR = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity\results2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset paths
DATA_PATHS = {
    "Chapman": r"D:\Engr GHANWA MS 220211001\Centralized_chapman_shaoxing_3_classes\pre_processed_data\chapman_preprocessed_4_labels.npz",
    "CPSC": r"D:\Engr GHANWA MS 220211001\Centralized_CPSC_3_classes\pre_processed_data\cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": r"D:\Engr GHANWA MS 220211001\Centralized_Georgia_3_classes\pre_processed_data\georgia_preprocessed_4_labels.npz",
    "Ningbo": r"D:\Engr GHANWA MS 220211001\Centralized_Ningbo_3_classes\pre_processed_data\ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": r"D:\Engr GHANWA MS 220211001\Centralized_Physionet_2017_4_classes\pre_processed_data\preprocessed_physionet2017_3class.npz",  
    "PTB": r"D:\Engr GHANWA MS 220211001\Centralized_PTB_3_classes\pre_processed_data\ptb_combined_preprocessed_4_labels.npz"
}

# =========================================================
# 🎨 Confusion Matrix + Report Saving
# =========================================================
def save_confusion_and_report(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    # ✅ Soft blue gradient (light to dark)
    colors = ["#EBF5FB", "#AED6F1", "#3498DB", "#1F618D"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=256)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=custom_cmap,
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 13, "weight": "bold", "color": "black"}
    )

    plt.title(f"Confusion Matrix - {name}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xticks(fontsize=11, rotation=30, ha='right')
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_confusion.png"), dpi=400, bbox_inches="tight")
    plt.close()

    # Save text report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(RESULTS_DIR, f"{name}_report.txt"), "w") as f:
        f.write(report)

# =========================================================
# 🧠 Evaluation Function
# =========================================================
def evaluate_model(model, dataloader, name):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            outputs = model(signals)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    save_confusion_and_report(y_true, y_pred, name)
    return weighted_f1, y_true, y_pred

# =========================================================
# 🚀 Main Evaluation Logic - Modified for equal-sample combined test
# =========================================================
def main():
    # Load data
    client_loaders = get_client_dataloaders(DATA_PATHS, BATCH_SIZE)

    # Load model
    model = CNN1DAttentionEnhanced(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Loaded model from: {MODEL_PATH}")

    all_y_true, all_y_pred = [], []
    f1_results = {}

    # Store test datasets temporarily for equal-sample combination
    test_subsets = []

    for client_name, loaders in client_loaders.items():
        print(f"\n🔍 Evaluating on {client_name} dataset...")

        # Validation set
        val_f1, _, _ = evaluate_model(model, loaders["val"], f"{client_name}_Validation")
        print(f"Validation Weighted F1 ({client_name}): {val_f1:.4f}")

        # Test set evaluation per client
        test_f1, y_true, y_pred = evaluate_model(model, loaders["test"], f"{client_name}_Test")
        print(f"Test Weighted F1 ({client_name}): {test_f1:.4f}")
        f1_results[client_name] = {"val_f1": val_f1, "test_f1": test_f1}

        # Store test samples for combined evaluation
        test_subsets.append((y_true, y_pred))

    # ---- Combine equal samples from each test set ----
    min_len = min(len(y) for y, _ in test_subsets)  # smallest test set length
    balanced_y_true, balanced_y_pred = [], []

    for y_true, y_pred in test_subsets:
        indices = np.random.choice(len(y_true), min_len, replace=False)
        balanced_y_true.extend(np.array(y_true)[indices])
        balanced_y_pred.extend(np.array(y_pred)[indices])

    # Combined evaluation across all datasets
    print("\n🌍 Evaluating combined balanced test set...")
    combined_f1 = f1_score(balanced_y_true, balanced_y_pred, average="weighted")
    save_confusion_and_report(balanced_y_true, balanced_y_pred, "Combined_Balanced_Test")
    print(f"Combined Balanced Test Weighted F1: {combined_f1:.4f}")

    # Save F1 summary
    f1_path = os.path.join(RESULTS_DIR, "weighted_f1_scores.txt")
    with open(f1_path, "w") as f:
        for name, scores in f1_results.items():
            f.write(f"{name} - Val F1: {scores['val_f1']:.4f}, Test F1: {scores['test_f1']:.4f}\n")
        f.write(f"\nCombined Balanced Test Weighted F1: {combined_f1:.4f}\n")

    print(f"\n✅ Evaluation complete. Results saved in:\n{RESULTS_DIR}")

# =========================================================
# 🏁 Run
# =========================================================
if __name__ == "__main__":
    main()
