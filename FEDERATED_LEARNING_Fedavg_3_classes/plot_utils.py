# plot_utils.py

import matplotlib.pyplot as plt
import os
from config import RESULTS_DIR

def plot_metrics(history, filename_prefix="fedavg"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Accuracy plot
    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    acc_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"📊 Plots saved to: {RESULTS_DIR}")
