# FedECG: A Lightweight Out-of-Distribution Federated Framework for Robust Heart Rhythm Classification in the Internet of Medical Things

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for reproducing the experimental results of the proposed **FedBN-based CNN-LSTM-Attention** federated learning framework for multi-center electrocardiogram (ECG) rhythm classification across heterogeneous clinical cohorts.

---

## Table of Contents
- [Overview](#overview)
- [Proposed Architecture](#proposed-architecture)
- [Dataset & Train/Val/Test Split](#dataset--trainvaltest-split)
- [Installation & Environment Setup](#installation--environment-setup)
- [Directory Structure](#directory-structure)
- [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
  - [1. Main Proposed Model (FedBN + CNN-LSTM-Attention)](#1-main-proposed-model-fedbn--cnn-lstm-attention)
  - [2. Out-of-Distribution (OOD / Leave-One-Hospital-Out) Cross-Validation](#2-out-of-distribution-ood--leave-one-hospital-out-cross-validation)
  - [3. Federated Strategy Comparison (FedAvg vs. FedProx vs. FedBN)](#3-federated-strategy-comparison-fedavg-vs-fedprox-vs-fedbn)
  - [4. Centralized & Isolated Single-Client Baselines](#4-centralized--isolated-single-client-baselines)
  - [5. Impact of Class Balancing (Balanced vs. Unbalanced)](#5-impact-of-class-balancing-balanced-vs-unbalanced)
  - [6. Hyperparameter & Architectural Ablation Studies](#6-hyperparameter--architectural-ablation-studies)
- [Summary of Key Reported Results](#summary-of-key-reported-results)
- [Citation](#citation)

---

## Overview

Clinical ECG classification across independent healthcare institutions faces two fundamental challenges:
1. **Data Privacy Restrictions:** Regulatory standards (HIPAA, GDPR) prevent aggregating raw patient data across hospitals.
2. **Inter-Institutional Heterogeneity (Non-IID):** Variations in recording hardware, lead configurations (single-lead vs. 12-lead), sampling rates, patient demographics, and arrhythmia prevalence cause severe domain shifts.

This repository implements the **FedBN-based CNN-LSTM-Attention** framework:
- **FedBN Aggregation:** Retains local Batch Normalization parameters on each hospital client while aggregating convolutional, recurrent, and attention weights on the central server, effectively mitigating client domain shifts.
- **Hybrid Feature Backbone:** 1D residual convolutional layers capture local morphology; a 2-layer LSTM models temporal dynamics; soft-attention highlights discriminative segments.
- **Edge Suitability:** Compact footprint (~**0.50M parameters**, **1.93 MB** storage), well-suited for edge and Internet of Medical Things (IoMT) hardware.

---

## Proposed Architecture

```
Input ECG Signal (12 leads × 5000 samples)
  │
  ├── Stage 1: Conv1D(12→16, k=9) + BatchNorm1D + ReLU + ResidualUnit(16) + MaxPool1D(2)
  ├── Stage 2: Conv1D(16→32, k=9) + BatchNorm1D + ReLU + ResidualUnit(32) + MaxPool1D(2)
  ├── Stage 3: Conv1D(32→64, k=9) + BatchNorm1D + ReLU + ResidualUnit(64) + MaxPool1D(2)
  ├── Stage 4: Conv1D(64→128, k=7) + BatchNorm1D + ReLU + ResidualUnit(128) + MaxPool1D(2)
  │
  ├── Permute to Temporal Dimension (Batch, Time, 128)
  ├── 2-Layer LSTM (Hidden size = 32, Dropout = 0.2)
  ├── Soft-Attention Layer (Attention over temporal representations)
  ├── Dense Layer (32 → 32) + ReLU + Dropout(0.5)
  └── Output Classifier (32 → 3 classes: Normal, AF, Other)
```

---

## Dataset & Train/Val/Test Split

The framework is evaluated across **six diverse, real-world ECG benchmark datasets**:

| Client ID | Dataset Name | Primary Lead Configuration | Native Sampling Rate |
|:---|:---|:---:|:---:|
| Client 1 | **Chapman-Shaoxing** | 12-lead | 500 Hz |
| Client 2 | **China Physiological Signal Challenge (CPSC-2018)** | 12-lead | 500 Hz |
| Client 3 | **Georgia 12-Lead ECG Challenge (G12EC)** | 12-lead | 500 Hz |
| Client 4 | **Ningbo First Hospital Database** | 12-lead | 500 Hz |
| Client 5 | **PhysioNet Challenge 2017** | Single-lead (lead I, converted to 12-lead) | 300 Hz → resampled to 500 Hz |
| Client 6 | **PTB-XL Diagnostic ECG Database** | 12-lead | 500 Hz |

### Important Note on Evaluation Integrity (Test Set)
> **All reported results (Accuracy, Weighted F1, Macro F1, Precision, Recall, AUC, AP) are evaluated strictly on the reserved 10% test set.**

Each client dataset is partitioned into:
- **80% Training:** Used strictly for client-side local optimization.
- **10% Validation:** Used strictly for checkpointing and hyperparameter validation.
- **10% Testing (Held-Out):** Kept completely isolated and evaluated only after federated training is concluded.

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/alirazi23/FedECG.git
cd FedECG
```

### 2. Create and Activate a Virtual Environment
```bash
# Using conda (recommended)
conda create -n fedecg python=3.10 -y
conda activate fedecg

# Or using venv
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install numpy pandas scikit-learn matplotlib
```

### 4. Data Placement
Download the preprocessed `.npz` dataset files and place them under the `data/` directory:
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

---

## Directory Structure

```
FedECG/
├── README.md                                    # Root reproduction instructions (this file)
├── FedECG_FedBN_LSTM/                           # Complete self-contained reproduction suite
│   ├── README.md                                # Sub-module documentation
│   ├── run_in_distribution_ood_fedbn_lstm.py   # Proposed FedBN: In-distribution & OOD benchmark
│   ├── run_federated_comparison.py              # Comparison: FedAvg vs. FedProx vs. FedBN
│   ├── run_unbalanced_fedbn_lstm.py             # Class balancing ablation
│   ├── run_centralized.py                       # Centralized training upper bound
│   ├── run_local_ablations.py                   # Isolated local training & parameter footprints
│   ├── run_architectural_ablation.py            # Architecture ablation (LSTM vs. GRU)
│   ├── run_ablation_lr_bs.py                    # Learning rate & batch size grid search
│   ├── run_fedadam_bn.py                        # FedAdam + FedBN implementation
│   ├── run_fedadam_baseline.py                  # Standard FedAdam baseline
│   └── results/                                 # Output CSV metrics and evaluation summaries
├── FEDERATED_LEARNING_Fedavg_3_classes/         # Standalone FedAvg baseline pipelines
├── scripts/                                     # Data preprocessing and utility scripts
└── models.py                                    # Model definitions
```

---

## Step-by-Step Reproduction Guide

All reproduction commands can be executed from the `FedECG_FedBN_LSTM` directory:

```bash
cd FedECG_FedBN_LSTM
```

### 1. Main Proposed Model (FedBN + CNN-LSTM-Attention)
To run the proposed **FedBN** framework over 50 communication rounds on balanced datasets:
```bash
python run_in_distribution_ood_fedbn_lstm.py
```
- **What it does:** Runs federated aggregation where local BN layers remain on each client and non-BN weights are averaged.
- **Evaluation:** Evaluates the converged global model on each client's held-out **10% test set**.
- **Outputs:** In-distribution test metrics (Accuracy, Macro F1, Weighted F1) and per-hospital breakdowns.

### 2. Out-of-Distribution (OOD / Leave-One-Hospital-Out) Cross-Validation
The same master script executes the **Leave-One-Hospital-Out** cross-validation benchmark:
- Trains on 5 hospital clients.
- Tests zero-shot generalization on the 6th, completely unseen hospital's **10% test set**.
- Iterates across all 6 clients as the held-out institution.

### 3. Federated Strategy Comparison (FedAvg vs. FedProx vs. FedBN)
To compare federated aggregation strategies under identical communication rounds (50 rounds) and identical client data:
```bash
python run_federated_comparison.py
```
- **Evaluated Strategies:**
  - `FedAvg`: Standard federated averaging (aggregates all weights including BN).
  - `FedProx`: Federated averaging with proximal term ($\mu = 0.01$) to handle system heterogeneity.
  - `FedBN` (Proposed): Local client Batch Normalization with server aggregation of non-BN layers.
- **Output:** Stored in `results/federated_strategy_comparison.csv`.

### 4. Centralized & Isolated Single-Client Baselines
To train an isolated model at each individual hospital without federation, and to run centralized training:
```bash
# Isolated client baselines and parameter footprint
python run_local_ablations.py

# Centralized upper-bound model (all 6 training sets pooled)
python run_centralized.py
```
- **Output:** Measures performance drops from training locally in isolation versus federated collaboration.

### 5. Impact of Class Balancing (Balanced vs. Unbalanced)
To demonstrate the necessity of multi-filter data balancing on non-IID client distributions:
```bash
python run_unbalanced_fedbn_lstm.py
```
- **What it does:** Trains FedBN on the raw unbalanced clinical distributions.
- **Comparison:** Demonstrates a **+3.4%** macro F1 gain when balanced using BioSPPy, NeuroKit2, and EngzeeMod filters.

### 6. Hyperparameter & Architectural Ablation Studies
```bash
# Hyperparameter grid search: Learning Rate ∈ {1e-3, 2e-3, 3e-3} × Batch Size ∈ {32, 64, 128}
python run_ablation_lr_bs.py

# Architectural comparison: Proposed LSTM vs. baseline GRU (hidden=64)
python run_architectural_ablation.py
```

---

## Summary of Key Reported Results

All metrics reported below were computed on the **held-out 10% test sets**:

### 1. In-Distribution Multi-Center Performance (50 Communication Rounds)

| Client Cohort | Baseline (FedAvg + GRU) | Architecture (FedAvg + LSTM) | Proposed Framework (**FedBN + LSTM**) | Improvement vs. Baseline |
|:---|:---:|:---:|:---:|:---:|
| **Chapman** | 0.9544 | 0.9594 | **0.9864** | +3.20% |
| **CPSC** | 0.7300 | 0.7590 | **0.8631** | **+13.31%** |
| **Georgia** | 0.7928 | 0.7867 | **0.8409** | +4.81% |
| **Ningbo** | 0.9231 | 0.9255 | **0.9485** | +2.54% |
| **PhysioNet 2017** | 0.6087 | 0.6532 | **0.7339** | **+12.52%** |
| **PTB** | 0.7310 | 0.7623 | **0.8704** | **+13.94%** |
| **Overall Average** | **0.7900 (79.00%)** | **0.8077 (80.77%)** | **0.8739 (87.39%)** | **+8.39%** |

### 2. Strategy Comparison (Global Test Set)

| Strategy | Backbone | Global Test Accuracy | Global Weighted F1 | Global Macro F1 |
|:---|:---|:---:|:---:|:---:|
| **FedAvg** | CNN-LSTM-Attention | 84.82% | 0.8478 | 0.8412 |
| **FedProx** ($\mu=0.01$) | CNN-LSTM-Attention | 83.91% | 0.8385 | 0.8310 |
| **FedBN (Proposed)** | **CNN-LSTM-Attention** | **86.86%** | **0.8688** | **0.8619** |
| *Centralized (Upper Bound)* | CNN-LSTM-Attention | 88.24% | 0.8816 | 0.8790 |

### 3. Edge / IoMT Computational Complexity

| Metric | Baseline ResNet-18 | Proposed CNN-LSTM-Attention | Reduction |
|:---|:---:|:---:|:---:|
| **Total Parameters** | ~11.17M | **0.50M** | **95.5% reduction** |
| **Model Size on Disk** | ~45.0 MB | **1.93 MB** | **95.7% reduction** |
| **Per-Round Uplink Payload** | ~44.7 MB | **1.93 MB** | **95.7% reduction** |
| **Theoretical FLOPs** | ~1800 MFLOPs | **~53.2 MFLOPs** | **97.0% reduction** |

---

## Verification & Sanity Check

To quickly verify that the environment and model checkpointing work without errors:
```bash
python -c "
import torch
from run_fedadam_bn import CNN1DAttentionEnhancedLSTM
model = CNN1DAttentionEnhancedLSTM(num_classes=3)
x = torch.randn(2, 12, 5000)
out = model(x)
print('Output shape:', out.shape)
assert out.shape == (2, 3), 'Shape mismatch'
print('Model verification successful!')
"
```

---

## Citation

If you find this repository, code, or experimental benchmarks useful in your research, please cite:

```bibtex
@article{fedecg2026plos,
  title={A Lightweight Out-of-Distribution Federated Framework for Robust Heart Rhythm Classification in the Internet of Medical Things},
  author={Ghanwa, Muhammad Awais and Razi, Ali},
  journal={PLOS ONE},
  year={2026},
  note={Manuscript ID: PONE-D-26-24560}
}
```
