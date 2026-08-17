"""PyTorch datasets for training."""

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, s, v, x, device):
        self.s = torch.tensor(s, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return (self.s[idx], self.x[idx]), self.v[idx]
