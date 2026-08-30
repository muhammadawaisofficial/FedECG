# evaluate_fedbn.py
"""
Evaluation for the FedBN checkpoint.

Because BatchNorm stats are private to each client under FedBN, there is no
single "global model" to evaluate. Instead, for each client we rebuild:

    shared (aggregated) backbone + THAT client's own local BatchNorm stats

and evaluate it on that client's own val/test split — the same per-client
val/test structure federated_train_fedbn.py trained with. A client that did
not take part in FedBN training has no personalized BN and is skipped (the
FedBN paper does not define a mechanism to serve unseen clients).
"""
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from config import DEVICE, RESULTS_DIR, SAVED_MODELS_DIR, NUM_CLASSES, INPUT_LENGTH
from models import CNN1DAttentionEnhanced


def evaluate_fedbn_model_on_clients(client_loaders, checkpoint_name="best_fedbn_model.pth"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ckpt = torch.load(os.path.join(SAVED_MODELS_DIR, checkpoint_name), map_location=DEVICE)
    non_bn_state = ckpt["non_bn_state"]
    client_bn_states = ckpt["client_bn_states"]
    bn_keys = ckpt["bn_keys"]
    non_bn_keys = ckpt["non_bn_keys"]

    f1_scores_log = []
    class_names = ["Normal", "AFIB", "Other Rhythm"]

    for client_name, loaders in client_loaders.items():
        if client_name not in client_bn_states:
            print(f"⚠️  No personalized BatchNorm stats for '{client_name}' "
                  f"(it did not take part in FedBN training) — skipping.")
            continue

        # Rebuild this client's personalized model.
        model = CNN1DAttentionEnhanced(num_classes=NUM_CLASSES, input_length=INPUT_LENGTH).to(DEVICE)
        state = model.state_dict()
        for k in non_bn_keys:
            state[k] = non_bn_state[k].clone()
        for k in bn_keys:
            state[k] = client_bn_states[client_name][k].clone()
        model.load_state_dict(state)
        model.eval()

        for split in ["val", "test"]:
            dataloader = loaders[split]
            all_preds, all_labels = [], []

            with torch.no_grad():
                for x, y in dataloader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = model(x)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            f1_scores_log.append(f"{client_name} - {split} Weighted F1: {f1:.4f}")

            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title(f"[FedBN] {client_name} - {split} Confusion Matrix")
            cm_path = os.path.join(RESULTS_DIR, f"fedbn_{client_name}_{split}_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

            report_path = os.path.join(RESULTS_DIR, f"fedbn_{client_name}_{split}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"[FedBN] Classification Report - {client_name} ({split})\n\n")
                f.write(report)

    with open(os.path.join(RESULTS_DIR, "fedbn_f1_scores_summary.txt"), "w") as f:
        for line in f1_scores_log:
            f.write(line + "\n")

    print(f"\n✅ FedBN evaluation complete. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    from config import DATA_PATHS, BATCH_SIZE
    from dataset import get_client_dataloaders

    client_loaders = get_client_dataloaders(DATA_PATHS, BATCH_SIZE)
    evaluate_fedbn_model_on_clients(client_loaders)
