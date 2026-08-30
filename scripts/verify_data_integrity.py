"""
Dataset Verification and Integrity Inspection Script
Verifies array shapes, sampling dimensions, class balances, and absence of NaNs.
"""

import os
import numpy as np
import pandas as pd

BALANCED_DIR = os.path.abspath("./data/balanced")
UNBALANCED_DIR = os.path.abspath("./data/unbalanced")

BALANCED_FILES = {
    "Chapman": "chapman_4_classes_balanced.npz",
    "CPSC": "cpsc_clean.npz",
    "Georgia": "georgia_clean.npz",
    "Ningbo": "ningbo_clean.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

UNBALANCED_FILES = {
    "Chapman": "chapman_preprocessed_4_labels.npz",
    "CPSC": "cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": "georgia_preprocessed_4_labels.npz",
    "Ningbo": "ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

def inspect_dataset_group(name, file_map, base_dir):
    print("\n" + "=" * 85)
    print(f"🔍 INSPECTING DATASET GROUP: {name.upper()} ({base_dir})")
    print("=" * 85)
    
    rows = []
    for client, filename in file_map.items():
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ Missing file: {filepath}")
            continue
            
        data = np.load(filepath)
        x_tr, y_tr = data['X_train'], data['y_train']
        x_va, y_va = data['X_val'], data['y_val']
        x_te, y_te = data['X_test'], data['y_test']
        
        # Check class distribution
        unique, counts = np.unique(y_tr, return_counts=True)
        dist = dict(zip(unique, counts))
        
        # Check shapes & NaNs
        has_nan = np.isnan(x_tr).any() or np.isnan(x_va).any() or np.isnan(x_te).any()
        
        rows.append({
            "Hospital": client,
            "Train Shape": f"{x_tr.shape}",
            "Val Shape": f"{x_va.shape}",
            "Test Shape": f"{x_te.shape}",
            "Total Samples": len(x_tr) + len(x_va) + len(x_te),
            "Train Class Balance": str(dist),
            "NaN Free": "✅ YES" if not has_nan else "❌ NO"
        })
        
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

def main():
    inspect_dataset_group("Balanced Benchmarks (Phase 1 Target)", BALANCED_FILES, BALANCED_DIR)
    if os.path.exists(UNBALANCED_DIR) and len(os.listdir(UNBALANCED_DIR)) > 0:
        inspect_dataset_group("Unbalanced Benchmarks", UNBALANCED_FILES, UNBALANCED_DIR)

if __name__ == "__main__":
    main()
