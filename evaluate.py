# evaluate.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from config import DEVICE, RESULTS_DIR, SAVED_MODELS_DIR, NUM_CLASSES, INPUT_LENGTH
from models import CNN1DAttentionEnhanced  # ✅ use 'model', not 'models'


def evaluate_saved_model_on_clients(client_loaders, model_path="best_federated_model.pth"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model = CNN1DAttentionEnhanced(num_classes=NUM_CLASSES, input_length=INPUT_LENGTH).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, model_path), map_location=DEVICE))
    model.eval()

    f1_scores_log = []
    class_names = ["Normal", "AFIB", "Other Rhythm"]

    for client_name, loaders in client_loaders.items():
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

            # Classification Report
            report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            f1_scores_log.append(f"{client_name} - {split} Weighted F1: {f1:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title(f"{client_name} - {split} Confusion Matrix")
            cm_path = os.path.join(RESULTS_DIR, f"{client_name}_{split}_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

            # Classification Report File
            report_path = os.path.join(RESULTS_DIR, f"{client_name}_{split}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Classification Report - {client_name} ({split})\n\n")
                f.write(report)

    # Save F1 Summary
    with open(os.path.join(RESULTS_DIR, "f1_scores_summary.txt"), "w") as f:
        for line in f1_scores_log:
            f.write(line + "\n")

    print(f"\n✅ Evaluation complete. Results saved to: {RESULTS_DIR}")


