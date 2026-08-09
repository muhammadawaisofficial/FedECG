import torch
import torch.nn as nn
import copy
import os
import numpy as np
import matplotlib.pyplot as plt

from models import CNN1DAttentionEnhanced
from config import (
    DEVICE, LEARNING_RATE, BATCH_SIZE, NUM_CLASSES,
    SAVED_MODELS_DIR, LOGS_DIR, LOCAL_EPOCHS, INPUT_LENGTH
)
from dataset import get_client_dataloaders


# ================================
# 🔹 LOCAL TRAINING (UNCHANGED)
# ================================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def average_weights(local_weights):
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    return avg_weights


# ================================
# 🔹 FEDERATED TRAINING (FedAdam)
# ================================
def federated_training(dataset_paths, num_global_epochs=50, local_epochs=LOCAL_EPOCHS):
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    global_model = CNN1DAttentionEnhanced(
        num_classes=NUM_CLASSES, input_length=INPUT_LENGTH
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    client_loaders = get_client_dataloaders(dataset_paths, batch_size=BATCH_SIZE)

    # 🔹 FedAdam hyperparameters (SERVER SIDE)
    server_lr = LEARNING_RATE
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
    v = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
    t = 0  # Adam timestep

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": None
    }

    best_model = None
    best_val_acc = 0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_global_epochs):
        print(f"\n🌍 Global Epoch {epoch + 1}/{num_global_epochs}")
        t += 1

        local_weights = []
        local_train_losses = []
        local_train_accuracies = []
        local_val_losses = []
        local_val_accuracies = []

        global_weights_prev = copy.deepcopy(global_model.state_dict())

        # -------- CLIENT TRAINING --------
        for client_name, loaders in client_loaders.items():
            print(f"  🧠 Client: {client_name}")

            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

            train_loader = loaders["train"]
            val_loader = loaders["val"]

            for local_epoch in range(local_epochs):
                train_loss, train_acc = train_one_epoch(
                    local_model, train_loader, criterion, optimizer
                )
                print(f"    🌀 Local Epoch {local_epoch+1}/{local_epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate(local_model, val_loader, criterion)
            print(f"    📊 Final Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_train_losses.append(train_loss)
            local_train_accuracies.append(train_acc)
            local_val_losses.append(val_loss)
            local_val_accuracies.append(val_acc)

        # -------- FedAvg (mean client update) --------
        avg_weights = average_weights(local_weights)

        # -------- FedAdam SERVER UPDATE --------
        with torch.no_grad():
            for key in global_weights_prev.keys():
                delta = avg_weights[key] - global_weights_prev[key]

                m[key] = beta1 * m[key] + (1 - beta1) * delta
                v[key] = beta2 * v[key] + (1 - beta2) * (delta ** 2)

                m_hat = m[key] / (1 - beta1 ** t)
                v_hat = v[key] / (1 - beta2 ** t)

                avg_weights[key] = global_weights_prev[key] + (
                    server_lr * m_hat / (torch.sqrt(v_hat) + epsilon)
                )

        global_model.load_state_dict(avg_weights)

        avg_train_loss = np.mean(local_train_losses)
        avg_train_acc = np.mean(local_train_accuracies)
        avg_val_loss = np.mean(local_val_losses)
        avg_val_acc = np.mean(local_val_accuracies)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(f"📌 Global Epoch {epoch+1} Summary:")
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

            torch.save(
                best_model,
                os.path.join(SAVED_MODELS_DIR, "best_fedadam_model.pth")
            )

            plt.figure()
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss")
            plt.legend()
            plt.title("FedAdam Loss Curve")
            plt.savefig(os.path.join(LOGS_DIR, "best_loss_curve.png"))
            plt.close()

            plt.figure()
            plt.plot(history["train_acc"], label="Train Acc")
            plt.plot(history["val_acc"], label="Val Acc")
            plt.legend()
            plt.title("FedAdam Accuracy Curve")
            plt.savefig(os.path.join(LOGS_DIR, "best_accuracy_curve.png"))
            plt.close()

            print(f"💾 Best FedAdam model + plots saved at epoch {epoch+1}")

    torch.save(
        global_model.state_dict(),
        os.path.join(SAVED_MODELS_DIR, "fedadam_final_global_model.pth")
    )

    history["best_epoch"] = best_epoch
    np.save(os.path.join(LOGS_DIR, "fedadam_training_history.npy"), history)

    return global_model, best_model, client_loaders, history
