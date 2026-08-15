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


def federated_training(dataset_paths, num_global_epochs=50, local_epochs=LOCAL_EPOCHS):
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    global_model = CNN1DAttentionEnhanced(
        num_classes=NUM_CLASSES, input_length=INPUT_LENGTH
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    client_loaders = get_client_dataloaders(dataset_paths, batch_size=BATCH_SIZE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": None
    }

    best_model = None
    best_val_acc = 0
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_global_epochs):
        print(f"\n🌍 Global Epoch {epoch + 1}/{num_global_epochs}")
        global_model.train()

        local_weights = []
        local_train_losses = []
        local_train_accuracies = []
        local_val_losses = []
        local_val_accuracies = []

        for client_name, loaders in client_loaders.items():
            print(f"  🧠 Client: {client_name}")
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

            train_loader = loaders["train"]
            val_loader = loaders["val"]

            for local_epoch in range(local_epochs):
                train_loss, train_acc = train_one_epoch(local_model, train_loader, criterion, optimizer)
                print(f"    🌀 Local Epoch {local_epoch+1}/{local_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate(local_model, val_loader, criterion)
            print(f"    📊 Final Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_train_losses.append(train_loss)
            local_train_accuracies.append(train_acc)
            local_val_losses.append(val_loss)
            local_val_accuracies.append(val_acc)

        # Federated averaging
        global_model.load_state_dict(average_weights(local_weights))

        # Logging averaged metrics
        avg_train_loss = np.mean(local_train_losses)
        avg_train_acc = np.mean(local_train_accuracies)
        avg_val_loss = np.mean(local_val_losses)
        avg_val_acc = np.mean(local_val_accuracies)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        # Print summary
        print(f"📌 Global Epoch {epoch+1} Summary:")
        print(f"   🔢 Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   📊 Avg Train Accuracy: {avg_train_acc:.4f}")
        print(f"   🔍 Avg Val Loss: {avg_val_loss:.4f}")
        print(f"   ✅ Avg Val Accuracy: {avg_val_acc:.4f}")

        # Save the best model (based on val acc, then val loss)
        if (avg_val_acc > best_val_acc) or (avg_val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = avg_val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(global_model.state_dict())
            torch.save(best_model, os.path.join(SAVED_MODELS_DIR, "best_federated_model.pth"))
            print(f"💾 Best model updated and saved at epoch {epoch+1}")

    # Save final global model
    final_model_path = os.path.join(SAVED_MODELS_DIR, "fedavg_global_model.pth")
    torch.save(global_model.state_dict(), final_model_path)
    print(f"\n✅ Final global model saved at: {final_model_path}")

    # Store best epoch info in history
    history["best_epoch"] = best_epoch

    # Save history to disk for plotting later
    history_path = os.path.join(LOGS_DIR, "federated_training_history.npy")
    np.save(history_path, history)
    print(f"🧾 Training history saved to: {history_path}")
    print(f"⭐ Best epoch = {best_epoch + 1} | Best Val Acc = {best_val_acc:.4f} | Best Val Loss = {best_val_loss:.4f}")

    # Optional: plot training and validation curves with best epoch marked
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel("Global Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Federated Training Accuracy Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel("Global Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Federated Training Loss Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "loss_curve.png"))
    plt.close()

    print(f"📈 Plots saved to {LOGS_DIR}/accuracy_curve.png and loss_curve.png")

    # Return both models and history
    return global_model, best_model, client_loaders, history
