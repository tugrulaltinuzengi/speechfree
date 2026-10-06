import torch
import numpy as np
from torch.utils.data import Dataset


class ElectrolarynxDataset(Dataset):
    def __init__(self, X, y, is_train=True, num_channels=3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train
        self.num_channels = num_channels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.is_train:
            x = self._augment(x)
        return x, y

    def _augment(self, x):
        # Channel dropout
        if torch.rand(1).item() < 0.1:
            di = torch.randint(0, self.num_channels, (1,)).item()
            x[di, :] = 0.0
        # Gaussian noise
        x = x + torch.randn_like(x) * 0.01
        # Time shift ±50 sample
        shift = torch.randint(-50, 51, (1,)).item()
        if shift != 0:
            xs = torch.zeros_like(x)
            if shift > 0: xs[:, shift:] = x[:, :-shift]
            else:        xs[:, :shift] = x[:, -shift:]
            x = xs
        # Magnitude scaling ±10%
        x = x * (1.0 + (torch.rand(1).item() - 0.5) * 0.2)
        return x
