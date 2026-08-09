# main.py

from config import DATA_PATHS, SAVED_MODELS_DIR, NUM_CLASSES
from federated_train import federated_training
from plot_utils import plot_metrics
from evaluate import evaluate_saved_model_on_clients
import os


def main():
    print(f"🚀 Starting Federated Training for {NUM_CLASSES} classes...")
    global_model, client_loaders, history = federated_training(DATA_PATHS)

    print("📈 Plotting training/validation metrics...")
    plot_metrics(history, filename_prefix="fedavg")  # Saves in 'plots/fedavg_metrics.png'

    print("🧪 Evaluating saved global model on each client's val/test sets...")
    model_path = os.path.join(SAVED_MODELS_DIR, "best_federated_model.pth")  # ✅ consistent with training
    evaluate_saved_model_on_clients(client_loaders, model_path=model_path)

    print("🎉 All tasks completed successfully!")


if __name__ == "__main__":
    main()
