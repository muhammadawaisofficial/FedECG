import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from models import CNN1DAttentionEnhanced  # assuming model.py file contains your CNN1DAttentionEnhanced class

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATHS = {
    "Chapman": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_chapman_shaoxing_3_classes_balanced\pre_processed_data\chapman_4_classes_balanced.npz",
    "CPSC": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_CPSC_3_classes_balanced\pre_processed_data\cpsc_clean.npz",
    "Georgia": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_Georgia_3_classes_balanced\pre_processed_data\georgia_clean.npz",
    "Ningbo": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_Ningbo_3_classes_balanced\pre_processed_data\ningbo_clean.npz",
    "PhysioNet2017": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_Physionet_2017_4_classes\pre_processed_data\preprocessed_physionet2017_3class.npz",
    "PTB": r"D:\Engr GHANWA MS 220211001\PREVIOUS RESULTS\Centralized_PTB_3_classes_balanced\pre_processed_data\ptb_clean.npz"
}

MODEL_PATH = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity_balanced_except_Georgia\saved_models\best_federated_model.pth"
RESULTS_DIR = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity_balanced_except_Georgia\results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
INPUT_LENGTH = 3100  # Adjust if your preprocessing changes this

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_data(npz_path):
    """Load train/val/test splits from an .npz file (handles upper/lowercase keys)."""
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    # Handle uppercase/lowercase naming automatically
    def get_key(possible_keys):
        for k in possible_keys:
            if k in keys:
                return k
        return None

    x_val_key = get_key(["x_val", "X_val"])
    y_val_key = get_key(["y_val", "Y_val"])
    x_test_key = get_key(["x_test", "X_test"])
    y_test_key = get_key(["y_test", "Y_test"])

    if not all([x_val_key, y_val_key, x_test_key, y_test_key]):
        raise KeyError(f"❌ Expected keys not found in {npz_path}. Found keys: {keys}")

    x_val = data[x_val_key]
    y_val = data[y_val_key]
    x_test = data[x_test_key]
    y_test = data[y_test_key]

    return (x_val, y_val), (x_test, y_test)


def evaluate_model(model, X, y):
    """Run model evaluation on given data arrays."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(X), 64):  # batch size = 64
            batch_x = torch.tensor(X[i:i+64], dtype=torch.float32).to(DEVICE)
            batch_y = torch.tensor(y[i:i+64], dtype=torch.long).to(DEVICE)

            outputs = model(batch_x)
            pred_labels = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            preds.extend(pred_labels.cpu().numpy())
            trues.extend(batch_y.cpu().numpy())
    return np.array(trues), np.array(preds)


def save_confusion_and_report(y_true, y_pred, dataset_name, split_name):
    """Save confusion matrix and classification report."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    # --- Save Confusion Matrix ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{dataset_name} - {split_name} Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{split_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # --- Save Classification Report ---
    report_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{split_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{dataset_name} - {split_name} Classification Report\n")
        f.write(report)
    
    print(f"✅ Saved results for {dataset_name} ({split_name})")
    return cm, report


# -----------------------------
# MAIN EVALUATION PIPELINE
# -----------------------------
def main():
    print("🚀 Loading global model...")
    model = CNN1DAttentionEnhanced(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_X_test, all_y_test = [], []

    for dataset_name, path in DATA_PATHS.items():
        print(f"\n📂 Evaluating on {dataset_name}...")

        (x_val, y_val), (x_test, y_test) = load_data(path)

        # Ensure correct shape (N, 12, L)
        if x_test.shape[1] != 12:
            x_test = np.transpose(x_test, (0, 2, 1))
            x_val = np.transpose(x_val, (0, 2, 1))

        # --- Validation Set ---
        y_true_val, y_pred_val = evaluate_model(model, x_val, y_val)
        save_confusion_and_report(y_true_val, y_pred_val, dataset_name, "Validation")

        # --- Test Set ---
        y_true_test, y_pred_test = evaluate_model(model, x_test, y_test)
        save_confusion_and_report(y_true_test, y_pred_test, dataset_name, "Test")

        # Collect for combined test
        all_X_test.append(x_test)
        all_y_test.append(y_test)

    # -----------------------------
    # COMBINED TEST EVALUATION
    # -----------------------------
    print("\n🔹 Evaluating on Combined Test Set...")
    X_combined = np.concatenate(all_X_test, axis=0)
    y_combined = np.concatenate(all_y_test, axis=0)

    y_true_combined, y_pred_combined = evaluate_model(model, X_combined, y_combined)
    save_confusion_and_report(y_true_combined, y_pred_combined, "Combined_All", "Test")

    print("\n✅ All evaluations completed. Results saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
