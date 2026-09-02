# FedECG: Federated Learning for Multi-Center ECG Classification
## Reproduction Code — FedBN + CNN-LSTM-Attention Architecture

This folder contains the complete, self-contained PyTorch scripts used to reproduce **every table and figure** in the PLOS ONE manuscript using the improved **FedBN + CNN-LSTM-Attention** architecture.

---

## Requirements
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- `pandas`, `numpy`, `scikit-learn`

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn
```

## Data Preparation & Train/Val/Test Split
Download the preprocessed `.npz` datasets from HuggingFace (`Alirazi/FedECG`) and place them in:
```
./data/balanced/       # Balanced datasets (Chapman, CPSC, Georgia, Ningbo, PhysioNet2017, PTB)
./data/Unbalanced/     # Unbalanced datasets (same 6 clients, original class distributions)
```

> **Evaluation Protocol:** All reported performance numbers are evaluated strictly on the **held-out 10% test set** of each client (split: 80% train, 10% val, 10% test). The test sets are never seen during federated training.

---

## Scripts → Paper Tables/Figures Mapping

| Script | Tables/Figures Reproduced | Description |
|--------|--------------------------|-------------|
| `run_in_distribution_ood_fedbn_lstm.py` | **Tables 1, 2, 5, 6** | Master 50-round FedBN training: In-Distribution benchmark + Leave-One-Hospital-Out (OOD) cross-validation |
| `run_unbalanced_fedbn_lstm.py` | **Table 6, Figure 6** | FedBN on unbalanced datasets to demonstrate the effect of class balancing |
| `run_local_ablations.py` | **Tables 3, 4, 9** | Single-client baselines, model parameter/size computation |
| `run_centralized.py` | **Table 3** | Centralized learning baseline — combines all 6 datasets into one model (upper bound) |
| `run_architectural_ablation.py` | **Table 11** | Compares proposed LSTM architecture vs baseline GRU (hidden=64, no recurrent dropout) |
| `run_federated_comparison.py` | **Table 10** | Head-to-head comparison of FedAvg vs FedProx vs FedBN strategies (50 rounds each) |
| `run_ablation_lr_bs.py` | **Table 8** | Learning Rate × Batch Size grid search ablation (3×3 = 9 configurations) |

---

## Quick Start — Reproduce All Results

### Step 1: Main Federated Results (Tables 1, 2, 5, 6)
```bash
python run_in_distribution_ood_fedbn_lstm.py
```
This runs the full 50-round OOD benchmark (Leave-One-Hospital-Out) followed by the In-Distribution baseline using balanced datasets. Results are saved to `./results/`.

### Step 2: Unbalanced Comparison (Table 6, Figure 6)
```bash
python run_unbalanced_fedbn_lstm.py
```
Evaluates FedBN + LSTM on the original unbalanced datasets to demonstrate the effect of class balancing.

### Step 3: Centralized & Single-Client Baselines (Tables 3, 4, 9)
```bash
python run_local_ablations.py
```
Trains isolated single-client models, evaluates centralized performance, and computes the model footprint (0.50M parameters, 1.93 MB).

### Step 4: FL Strategy Comparison (Table 10)
```bash
python run_federated_comparison.py
```
Runs FedAvg, FedProx, and FedBN side-by-side for 50 communication rounds each and outputs a CSV comparison.

### Step 5: LR/BS Hyperparameter Ablation (Table 8)
```bash
python run_ablation_lr_bs.py
```
Grid search over LR ∈ {0.001, 0.002, 0.003} × BS ∈ {32, 64, 128}. Best config: LR=0.001, BS=128 (Val F1: 0.9051).

### Step 6: Architectural Ablation (Table 11)
```bash
python run_architectural_ablation.py
```
Evaluates a baseline GRU architecture to demonstrate the superiority of the proposed LSTM design.

---

## Model Architecture Summary
- **CNN-LSTM-Attention** (Proposed): 0.50M parameters, 1.93 MB
- 4-stage 1D CNN with Residual Blocks + BatchNorm
- 2-layer Bidirectional LSTM (hidden=128, recurrent dropout=0.3)
- Soft Attention mechanism
- FedBN: Client-local BatchNorm layers, global non-BN aggregation

## Key Results
| Metric | Score |
|--------|-------|
| FedBN Weighted F1 | **0.8688** |
| FedAvg Weighted F1 | 0.8478 |
| FedProx Weighted F1 | 0.8385 |
| Centralized Macro F1 | 0.8816 |

---

## Citation
If you use this code, please cite the corresponding PLOS ONE manuscript.
