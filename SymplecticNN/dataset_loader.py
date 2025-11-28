# Handles loading from lorenzSystemSimulator/data
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LorenzDataset(Dataset):

    def __init__(self, npyPath: str, normalize: bool = True):
        super().__init__()
        self.data = np.load(npyPath)  # shape (num_trajectories, time_steps, 3)
        self.normalize = normalize

        X = self.data[:, :-1, :] 
        Y = self.data[:, 1:, :]

        self.X = X.reshape(-1, 3) 
        self.Y = Y.reshape(-1, 3)

        if normalize:
            self._compute_norm_stats()
            self._apply_norm()

    def _compute_norm_stats(self):
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-8

    def _apply_norm(self):
        self.X = (self.X - self.mean) / self.std
        self.Y = (self.Y - self.mean) / self.std

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y
    
def get_dataloader(npyPath: str, batchSize: int = 256, shuffle: bool = True):
    dataset = LorenzDataset(npyPath)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
    return loader