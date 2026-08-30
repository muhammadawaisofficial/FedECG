# federated_train_fedavg.py
"""
Plain FedAvg (McMahan et al., 2017): the server update is a straight mean of
client weights — no server-side Adam/momentum state like federated_train.py's
FedAdam. This is what's needed to fairly evaluate "FedAvg with an LSTM
backbone instead of GRU": the aggregation rule (FedAvg) is held fixed while
the model architecture is swapped.

`model_class` is any callable with signature
    model_class(num_classes=..., input_length=...) -> nn.Module
so this same loop works for CNN1DAttentionEnhanced (GRU) and
CNN1DAttentionEnhancedLSTM (LSTM) without duplicating the training logic.

Re-uses train_one_epoch / evaluate / average_weights from federated_train.py
— average_weights there is already plain FedAvg, it's simply never called on
its own in that file because the FedAdam server step overwrites its output.
"""
import torch
import torch.nn as nn
import copy
import os
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DEVICE, LEARNING_RATE, BATCH_SIZE, NUM_CLASSES,
    SAVED_MODELS_DIR, LOGS_DIR, LOCAL_EPOCHS, INPUT_LENGTH
)
from dataset import get_client_dataloaders
from federated_train import train_one_epoch, evaluate, average_weights


def federated_training_fedavg(dataset_paths, model_class, num_global_epochs=50,
                               local_epochs=LOCAL_EPOCHS, model_tag="fedavg"):
    """
    model_class : e.g. models.CNN1DAttentionEnhanced or models.CNN1DAttentionEnhancedLSTM
    model_tag   : used to name saved checkpoints/plots, e.g. "fedavg_gru", "fedavg_lstm"
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    global_model = model_class(num_classes=NUM_CLASSES, input_length=INPUT_LENGTH).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    client_loaders = get_client_dataloaders(dataset_paths, batch_size=BATCH_SIZE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": None}

    best_model = None
    best_val_acc = 0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_global_epochs):
        print(f"\n🌍 [FedAvg:{model_tag}] Global Epoch {epoch + 1}/{num_global_epochs}")

        local_weights = []
        local_train_losses, local_train_accuracies = [], []
        local_val_losses, local_val_accuracies = [], []

        for client_name, loaders in client_loaders.items():
            print(f"  🧠 Client: {client_name}")

            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

            train_loader = loaders["train"]
            val_loader = loaders["val"]

            for local_epoch in range(local_epochs):
                train_loss, train_acc = train_one_epoch(local_model, train_loader, criterion, optimizer)
                print(f"    🌀 Local Epoch {local_epoch+1}/{local_epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate(local_model, val_loader, criterion)
            print(f"    📊 Final Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_train_losses.append(train_loss)
            local_train_accuracies.append(train_acc)
            local_val_losses.append(val_loss)
            local_val_accuracies.append(val_acc)

        # -------- Plain FedAvg aggregation (straight mean, NO server-side Adam) --------
        avg_weights = average_weights(local_weights)
        global_model.load_state_dict(avg_weights)

        avg_train_loss = np.mean(local_train_losses)
        avg_train_acc = np.mean(local_train_accuracies)
        avg_val_loss = np.mean(local_val_losses)
        avg_val_acc = np.mean(local_val_accuracies)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(f"📌 [FedAvg:{model_tag}] Global Epoch {epoch+1} Summary:")
        print(f"   🔢 Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   📊 Avg Train Accuracy: {avg_train_acc:.4f}")
        print(f"   🔍 Avg Val Loss: {avg_val_loss:.4f}")
        print(f"   ✅ Avg Val Accuracy: {avg_val_acc:.4f}")

        # -------- SAVE BEST MODEL --------
        if (avg_val_acc > best_val_acc) or (
            avg_val_acc == best_val_acc and avg_val_loss < best_val_loss
        ):
            best_val_acc = avg_val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(global_model.state_dict())

            torch.save(best_model, os.path.join(SAVED_MODELS_DIR, f"best_{model_tag}_model.pth"))

            plt.figure()
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss")
            plt.legend()
            plt.title(f"{model_tag} Loss Curve")
            plt.savefig(os.path.join(LOGS_DIR, f"{model_tag}_loss_curve.png"))
            plt.close()

            plt.figure()
            plt.plot(history["train_acc"], label="Train Acc")
            plt.plot(history["val_acc"], label="Val Acc")
            plt.legend()
            plt.title(f"{model_tag} Accuracy Curve")
            plt.savefig(os.path.join(LOGS_DIR, f"{model_tag}_accuracy_curve.png"))
            plt.close()

            print(f"💾 Best {model_tag} model + plots saved at epoch {epoch+1}")

    torch.save(
        global_model.state_dict(),
        os.path.join(SAVED_MODELS_DIR, f"{model_tag}_final_global_model.pth")
    )

    history["best_epoch"] = best_epoch
    np.save(os.path.join(LOGS_DIR, f"{model_tag}_training_history.npy"), history)

    return global_model, best_model, client_loaders, history
