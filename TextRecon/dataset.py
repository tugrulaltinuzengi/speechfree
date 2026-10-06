import torch
import numpy as np
from torch.utils.data import Dataset


class ElectrolarynxDataset(Dataset):
    def __init__(self, X, y, is_train=True):
        """
        X: (N, 5, 1000) numpy array <-- 5 Kanala düştü
        y: (N,) numpy array (labels 0-9)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx].clone()
        y_sample = self.y[idx]

        if self.is_train:
            x_sample = self._apply_augmentations(x_sample)

        return x_sample, y_sample

    def _apply_augmentations(self, x):
        # 1. Channel Dropout (p=0.1) - Rastgele bir kanalı sıfırla (0-4 arası)
        if torch.rand(1).item() < 0.1:
            drop_idx = torch.randint(0, 5, (1,)).item()  # <-- 6 yerine 5 yapıldı
            x[drop_idx, :] = 0.0

        # 2. Gaussian Noise (std=0.01)
        noise = torch.randn_like(x) * 0.01
        x = x + noise

        # 3. Random Time Shift (±50 sample) -> 100ms kayma toleransı
        shift = torch.randint(-50, 51, (1,)).item()
        if shift != 0:
            x_shifted = torch.zeros_like(x)
            if shift > 0:
                x_shifted[:, shift:] = x[:, :-shift]
            else:
                x_shifted[:, :shift] = x[:, -shift:]
            x = x_shifted

        return x