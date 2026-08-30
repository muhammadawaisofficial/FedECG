"""
High-Speed Multi-Threaded Dataset Ingestion Script
Fetches both Balanced and Unbalanced ECG benchmarks from Hugging Face in parallel.
"""

import os
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor

HF_TOKEN = "YOUR_HF_TOKEN"
REPO_ID = "Alirazi/FedECG"

BALANCED_DIR = os.path.abspath("./data/balanced")
UNBALANCED_DIR = os.path.abspath("./data/unbalanced")

os.makedirs(BALANCED_DIR, exist_ok=True)
os.makedirs(UNBALANCED_DIR, exist_ok=True)

BALANCED_FILES = {
    "Chapman": "chapman_4_classes_balanced.npz",
    "CPSC": "cpsc_clean.npz",
    "Georgia": "georgia_clean.npz",
    "Ningbo": "ningbo_clean.npz",
    "PhysioNet2017": "preprocessed_physionet2017_3class.npz",
    "PTB": "ptb_combined_preprocessed_4_labels.npz"
}

UNBALANCED_FILES = {
    "Chapman": "Unbalanced/chapman_preprocessed_4_labels.npz",
    "CPSC": "Unbalanced/cpsc_combined_preprocessed_4_labels.npz",
    "Georgia": "Unbalanced/georgia_preprocessed_4_labels.npz",
    "Ningbo": "Unbalanced/ningbo_preprocessed_4_labels.npz",
    "PhysioNet2017": "Unbalanced/preprocessed_physionet2017_3class.npz",
    "PTB": "Unbalanced/ptb_combined_preprocessed_4_labels.npz"
}

def download_file(item, target_dir):
    name, remote_filename = item
    dest_filename = os.path.basename(remote_filename)
    dest_path = os.path.join(target_dir, dest_filename)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        print(f"  [Found] {name} ({dest_filename}) already exists ({os.path.getsize(dest_path)/(1024*1024):.1f} MB).")
        return dest_path
    
    print(f"  [Downloading] {name} ({remote_filename})...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=remote_filename,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=target_dir
    )
    print(f"  [Completed] {name} -> {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
    return path

def main():
    print("=" * 70)
    print("🚀 HIGH-SPEED MULTI-THREADED DATASET INGESTION")
    print("=" * 70)
    
    print("\n📦 1. Downloading 6 Balanced Hospital Datasets (Parallel)...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(download_file, item, BALANCED_DIR) for item in BALANCED_FILES.items()]
        for f in futures: f.result()
        
    print("\n📦 2. Downloading 6 Unbalanced Hospital Datasets (Parallel)...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(download_file, item, UNBALANCED_DIR) for item in UNBALANCED_FILES.items()]
        for f in futures: f.result()

    print("\n" + "=" * 70)
    print("🎉 ALL BALANCED AND UNBALANCED DATASETS DOWNLOADED AND READY!")
    print(f"   Balanced Directory:   {BALANCED_DIR}")
    print(f"   Unbalanced Directory: {UNBALANCED_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
