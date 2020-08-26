# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.utils.data import Dataset


class VectorDataset(Dataset):
    def __init__(self, dimension, length, seed=None):
        self._length = length
        self._dim = dimension
        self._seed = seed
        np.random.seed(self._seed)
        self.vectors = [np.random.random((dimension, 1, 1))
                        for _ in range(length)]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx], dtype=torch.float32)
