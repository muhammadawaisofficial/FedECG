# federated_train_fedbn.py
"""
FedBN (Li et al., 2021 — "FedBN: Federated Learning on Non-IID Features via
Local Batch Normalization").

Core idea: every BatchNorm1d layer (weight, bias, running_mean, running_var,
num_batches_tracked) is kept 100% local to each client. Those parameters are
NEVER sent to the server and NEVER averaged. Every other parameter (conv
kernels, GRU/LSTM weights, attention, FC layers) is aggregated across clients
with plain FedAvg, exactly like federated_train.py's `average_weights`.

This directly targets the non-IID problem across the 6 ECG datasets (Chapman,
CPSC, Georgia, Ningbo, PhysioNet2017, PTB), since each site's signal
acquisition/preprocessing shifts the feature statistics that BatchNorm
normalizes — those shifts are exactly what FedBN lets each client absorb
locally instead of forcing one shared BN into an averaged compromise.

Re-uses train_one_epoch / evaluate from federated_train.py so the local
optimization loop is identical to the existing FedAdam pipeline; only the
aggregation rule differs.
"""
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
from federated_train import train_one_epoch, evaluate


def is_bn_key(key: str) -> bool:
    """BatchNorm1d params/buffers inside ConvBlock are named '*.bn.*'."""
    return ".bn." in key


def average_non_bn_weights(local_non_bn_weights):
    """Plain FedAvg mean, restricted to the non-BatchNorm keys already
    filtered out by the caller."""
    avg_weights = copy.deepcopy(local_non_bn_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_non_bn_weights)):
            avg_weights[key] = avg_weights[key] + local_non_bn_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_non_bn_weights))
    return avg_weights


def federated_training_fedbn(dataset_paths, num_global_epochs=50, local_epochs=LOCAL_EPOCHS):
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    template_model = CNN1DAttentionEnhanced(
        num_classes=NUM_CLASSES, input_length=INPUT_LENGTH
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    client_loaders = get_client_dataloaders(dataset_paths, batch_size=BATCH_SIZE)
    client_names = list(client_loaders.keys())

    full_state = template_model.state_dict()
    bn_keys = [k for k in full_state.keys() if is_bn_key(k)]
    non_bn_keys = [k for k in full_state.keys() if not is_bn_key(k)]
    print(f"🔎 FedBN: {len(bn_keys)} BatchNorm tensors kept local per client, "
          f"{len(non_bn_keys)} tensors aggregated globally.")

    # 🔹 Persistent per-client BatchNorm state — the heart of FedBN.
    # Initialized identically for every client, then diverges as each client trains.
    client_bn_states = {
        name: {k: full_state[k].clone() for k in bn_keys} for name in client_names
    }
    global_non_bn_state = {k: full_state[k].clone() for k in non_bn_keys}

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": None}

    best_val_acc = 0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_global_epochs):
        print(f"\n🌍 [FedBN] Global Epoch {epoch + 1}/{num_global_epochs}")

        local_non_bn_weights = []
        local_train_losses, local_train_accuracies = [], []
        local_val_losses, local_val_accuracies = [], []

        for client_name in client_names:
            loaders = client_loaders[client_name]
            print(f"  🧠 Client: {client_name}")

            # Build this round's local model: global non-BN backbone + THIS
            # client's own persisted BN stats (never the global average).
            local_model = CNN1DAttentionEnhanced(
                num_classes=NUM_CLASSES, input_length=INPUT_LENGTH
            ).to(DEVICE)
            local_state = local_model.state_dict()
            for k in non_bn_keys:
                local_state[k] = global_non_bn_state[k].clone()
            for k in bn_keys:
                local_state[k] = client_bn_states[client_name][k].clone()
            local_model.load_state_dict(local_state)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

            train_loader = loaders["train"]
            val_loader = loaders["val"]

            for local_epoch in range(local_epochs):
                train_loss, train_acc = train_one_epoch(local_model, train_loader, criterion, optimizer)
                print(f"    🌀 Local Epoch {local_epoch+1}/{local_epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate(local_model, val_loader, criterion)
            print(f"    📊 Final Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            updated_state = local_model.state_dict()

            # Save this client's BN stats back locally — they are NEVER uploaded.
            client_bn_states[client_name] = {k: updated_state[k].clone() for k in bn_keys}

            # Only the non-BN backbone goes to the server for aggregation.
            local_non_bn_weights.append({k: updated_state[k].clone() for k in non_bn_keys})

            local_train_losses.append(train_loss)
            local_train_accuracies.append(train_acc)
            local_val_losses.append(val_loss)
            local_val_accuracies.append(val_acc)

        # -------- FedAvg aggregation of the NON-BN backbone only (= FedBN) --------
        global_non_bn_state = average_non_bn_weights(local_non_bn_weights)

        avg_train_loss = np.mean(local_train_losses)
        avg_train_acc = np.mean(local_train_accuracies)
        avg_val_loss = np.mean(local_val_losses)
        avg_val_acc = np.mean(local_val_accuracies)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(f"📌 [FedBN] Global Epoch {epoch+1} Summary:")
        print(f"   🔢 Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   📊 Avg Train Accuracy: {avg_train_acc:.4f}")
        print(f"   🔍 Avg Val Loss: {avg_val_loss:.4f}")
        print(f"   ✅ Avg Val Accuracy: {avg_val_acc:.4f}")

        # -------- SAVE BEST CHECKPOINT --------
        if (avg_val_acc > best_val_acc) or (
            avg_val_acc == best_val_acc and avg_val_loss < best_val_loss
        ):
            best_val_acc = avg_val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch

            # There is no single "global BN" in FedBN, so the checkpoint stores
            # the shared backbone PLUS every client's private BN stats together.
            torch.save(
                {
                    "non_bn_state": copy.deepcopy(global_non_bn_state),
                    "client_bn_states": copy.deepcopy(client_bn_states),
                    "bn_keys": bn_keys,
                    "non_bn_keys": non_bn_keys,
                },
                os.path.join(SAVED_MODELS_DIR, "best_fedbn_model.pth")
            )

            plt.figure()
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss")
            plt.legend()
            plt.title("FedBN Loss Curve")
            plt.savefig(os.path.join(LOGS_DIR, "fedbn_loss_curve.png"))
            plt.close()

            plt.figure()
            plt.plot(history["train_acc"], label="Train Acc")
            plt.plot(history["val_acc"], label="Val Acc")
            plt.legend()
            plt.title("FedBN Accuracy Curve")
            plt.savefig(os.path.join(LOGS_DIR, "fedbn_accuracy_curve.png"))
            plt.close()

            print(f"💾 Best FedBN checkpoint + plots saved at epoch {epoch+1}")

    # Final (last-round) checkpoint too
    torch.save(
        {
            "non_bn_state": global_non_bn_state,
            "client_bn_states": client_bn_states,
            "bn_keys": bn_keys,
            "non_bn_keys": non_bn_keys,
        },
        os.path.join(SAVED_MODELS_DIR, "fedbn_final_model.pth")
    )

    history["best_epoch"] = best_epoch
    np.save(os.path.join(LOGS_DIR, "fedbn_training_history.npy"), history)

    return global_non_bn_state, client_bn_states, client_loaders, history
