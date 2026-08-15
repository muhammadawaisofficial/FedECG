# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return torch.tensor(signal), torch.tensor(label)


def load_npz_dataset(npz_path):
    data = np.load(npz_path)
    train_dataset = ECGDataset(data['X_train'], data['y_train'])
    val_dataset = ECGDataset(data['X_val'], data['y_val'])
    test_dataset = ECGDataset(data['X_test'], data['y_test'])
    return train_dataset, val_dataset, test_dataset


def get_client_dataloaders(dataset_paths, batch_size):
    client_loaders = {}
    for client_name, path in dataset_paths.items():
        train_set, val_set, test_set = load_npz_dataset(path)

        client_loaders[client_name] = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
        }

    return client_loaders
