import os
import torch

# 📦 Dataset paths (updated to 3-class versions)
DATA_PATHS = {
    "Chapman": r"D:\Engr GHANWA MS 220211001\Centralized_chapman_shaoxing_3_classes\pre_processed_data\chapman_preprocessed_4_labels.npz",
    "CPSC": r"D:\Engr GHANWA MS 220211001\Centralized_CPSC_3_classes\pre_processed_data\cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": r"D:\Engr GHANWA MS 220211001\Centralized_Georgia_3_classes\pre_processed_data\georgia_preprocessed_4_labels.npz",
    "Ningbo": r"D:\Engr GHANWA MS 220211001\Centralized_Ningbo_3_classes\pre_processed_data\ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": r"D:\Engr GHANWA MS 220211001\Centralized_Physionet_2017_4_classes\pre_processed_data\preprocessed_physionet2017_3class.npz",  
    "PTB": r"D:\Engr GHANWA MS 220211001\Centralized_PTB_3_classes\pre_processed_data\ptb_combined_preprocessed_4_labels.npz"
}

# 📁 Output directories
BASE_DIR = r"D:\Engr GHANWA MS 220211001\FEDERATED_LEARNING_Fedavg_3_classes_Reducing_complexity"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 🧠 Model & data settings
NUM_CLASSES = 3               # Match centralized setup (Normal, AFIB, Other)
NUM_LEADS = 12
INPUT_LENGTH = 3100

# 🏋️‍♂️ Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 50              # Number of global communication rounds
LOCAL_EPOCHS = 5             # Number of local training epochs per round
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15                # Early stopping patience (global)

# 💻 Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

