# FedECG: A Lightweight Out-of-Distribution Federated Framework for Robust Heart Rhythm Classification in the Internet of Medical Things

This repository contains the PyTorch implementation and scripts to reproduce all experiments and results in the paper:

> **A Lightweight Out-of-Distribution Federated Framework for Robust Heart Rhythm Classification in the Internet of Medical Things**  
> *PLOS ONE* (Manuscript ID: PONE-D-26-24560)

The proposed framework uses a **FedBN-based CNN-LSTM-Attention** architecture to perform multi-center ECG classification across six heterogeneous clinical cohorts while preserving institutional privacy.

---

## 1. Environment & Dependencies

Tested with Python 3.10 and PyTorch 2.x on Linux and Windows. GPU acceleration (CUDA) is recommended for training.

```bash
# Clone the repository
git clone https://github.com/alirazi23/FedECG.git
cd FedECG

# Install required packages
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

---

## 2. Dataset Setup & Partitioning

The experimental evaluation is conducted on six ECG benchmark datasets:
1. **Chapman-Shaoxing** (12-lead, 500 Hz)
2. **CPSC-2018** (12-lead, 500 Hz)
3. **Georgia / G12EC** (12-lead, 500 Hz)
4. **Ningbo** (12-lead, 500 Hz)
5. **PhysioNet 2017** (single-lead lead I converted to 12-lead format, resampled to 500 Hz)
6. **PTB-XL** (12-lead, 500 Hz)

Three clinical classes are targeted: **Normal Sinus Rhythm (NSR)**, **Atrial Fibrillation (AF)**, and **Other Arrhythmias**.

### Data Directory Structure
Place the preprocessed `.npz` files in the `data/` folder as follows:

```
data/
├── balanced/
│   ├── chapman_4_classes_balanced.npz
│   ├── cpsc_clean.npz
│   ├── georgia_clean.npz
│   ├── ningbo_clean.npz
│   ├── preprocessed_physionet2017_3class.npz
│   └── ptb_combined_preprocessed_4_labels.npz
└── Unbalanced/
    ├── chapman_unbalanced.npz
    ├── cpsc_unbalanced.npz
    ├── georgia_unbalanced.npz
    ├── ningbo_unbalanced.npz
    ├── physionet2017_unbalanced.npz
    └── ptb_unbalanced.npz
```

### Train / Validation / Test Split
Each client dataset is partitioned into:
- **80% Training:** Used strictly for local model updates on each client.
- **10% Validation:** Used for monitoring loss and learning rate scheduling.
- **10% Testing:** Completely held out and reserved exclusively for final evaluation.

> **Note:** All reported performance numbers (Accuracy, F1 scores, AUC, AP, confusion matrices) in the paper are evaluated strictly on the **10% held-out test sets**.

---

## 3. Proposed Model Architecture (CNN-LSTM-Attention)

The proposed network is designed to be lightweight for edge / IoMT deployment:
- **Feature Extractor:** 4-stage 1D Convolutional blocks with residual units (filters: 16, 32, 64, 128) and batch normalization.
- **Temporal Modeling:** 2-layer LSTM (hidden dimension: 32, recurrent dropout: 0.2).
- **Attention Mechanism:** Soft-attention over temporal slices to weight informative segments.
- **Classifier:** Fully connected layers (32 → 32 → 3 classes) with dropout (0.5).
- **Model Footprint:** ~**0.50M parameters** (**1.93 MB**), providing a ~95% reduction in size compared to standard ResNet-18 architectures.

### Federated Aggregation (FedBN)
Under cross-institutional heterogeneity, client-specific batch normalization parameters ($\gamma, \beta$, running mean/variance) are kept local to each client. Only non-BN weights (convolutions, residual connections, LSTM, attention, and linear heads) are transmitted and averaged by the central server.

---

## 4. Reproducing Experimental Results

All reproduction scripts are located in the `FedECG_FedBN_LSTM/` directory. Change to this directory before running:

```bash
cd FedECG_FedBN_LSTM
```

### (A) Main Proposed Model — In-Distribution & OOD Benchmark
To run the full 50-round FedBN training and evaluate both In-Distribution and Leave-One-Hospital-Out Out-of-Distribution (OOD) performance:
```bash
python run_in_distribution_ood_fedbn_lstm.py
```
This script evaluates the converged global model on the isolated test set of each client, producing the metrics reported in **Tables 1, 2, 5, and 6**.

### (B) Strategy Comparison (FedAvg vs. FedProx vs. FedBN)
To evaluate the impact of different federated aggregation algorithms under identical client data and round budgets (50 rounds):
```bash
python run_federated_comparison.py
```
This produces side-by-side test evaluations for standard FedAvg, FedProx ($\mu=0.01$), and FedBN (**Table 10**).

### (C) Effect of Dataset Balancing (Balanced vs. Unbalanced)
To train the FedBN framework on raw unbalanced client distributions and evaluate the contribution of signal-level data balancing:
```bash
python run_unbalanced_fedbn_lstm.py
```
This reproduces the comparison reported in **Table 6** and **Figure 6**.

### (D) Centralized & Single-Client Baselines
To evaluate performance when each hospital trains only on its own local data without federation, as well as the centralized upper bound:
```bash
# Local isolated training & parameter counting (Tables 3, 4, 9)
python run_local_ablations.py

# Centralized pooled training (Table 3)
python run_centralized.py
```

### (E) Hyperparameter & Architectural Ablations
```bash
# Learning rate & batch size grid search (Table 8)
python run_ablation_lr_bs.py

# Architecture comparison: proposed LSTM vs. baseline GRU (Table 11)
python run_architectural_ablation.py
```

---

## 5. Summary of Key Results (Evaluated on Held-Out Test Sets)

### In-Distribution Client Performance (50 Rounds, Balanced Data)

| Client / Cohort | FedAvg + GRU (Baseline) | FedAvg + LSTM | **FedBN + LSTM (Proposed)** | Gain vs. Baseline |
|:---|:---:|:---:|:---:|:---:|
| Chapman | 0.9544 | 0.9594 | **0.9864** | +3.20% |
| CPSC | 0.7300 | 0.7590 | **0.8631** | +13.31% |
| Georgia | 0.7928 | 0.7867 | **0.8409** | +4.81% |
| Ningbo | 0.9231 | 0.9255 | **0.9485** | +2.54% |
| PhysioNet 2017 | 0.6087 | 0.6532 | **0.7339** | +12.52% |
| PTB | 0.7310 | 0.7623 | **0.8704** | +13.94% |
| **Average** | **0.7900** | **0.8077** | **0.8739** | **+8.39%** |

### Federated Strategy Comparison (Global Test Set)

| Strategy | Backbone Model | Global Accuracy | Weighted F1 | Macro F1 |
|:---|:---|:---:|:---:|:---:|
| FedAvg | CNN-LSTM-Attention | 84.82% | 0.8478 | 0.8412 |
| FedProx ($\mu=0.01$) | CNN-LSTM-Attention | 83.91% | 0.8385 | 0.8310 |
| **FedBN (Proposed)** | **CNN-LSTM-Attention** | **86.86%** | **0.8688** | **0.8619** |
| Centralized (Upper Bound) | CNN-LSTM-Attention | 88.24% | 0.8816 | 0.8790 |

### Computational Complexity & IoMT Suitability

| Architecture | Total Parameters | Model Size | Uplink / Round | MFLOPs |
|:---|:---:|:---:|:---:|:---:|
| ResNet-18 Baseline | ~11.17M | ~45.0 MB | ~44.7 MB | ~1800 |
| **Proposed CNN-LSTM-Attention** | **0.50M** | **1.93 MB** | **1.93 MB** | **~53.2** |
| *Reduction* | *95.5%* | *95.7%* | *95.7%* | *97.0%* |

---

## Citation

```bibtex
@article{fedecg2026plos,
  title={A Lightweight Out-of-Distribution Federated Framework for Robust Heart Rhythm Classification in the Internet of Medical Things},
  author={Ghanwa, Muhammad Awais and Razi, Ali},
  journal={PLOS ONE},
  year={2026},
  note={Manuscript ID: PONE-D-26-24560}
}
```
