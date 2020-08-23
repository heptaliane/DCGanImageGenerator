# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.utils.data import IterableDataset


class VectorDataset(IterableDataset):
    def __init__(self, dimension, length, seed=None):
        self._length = length
        self._dim = dimension
        self._seed = seed
        self._reset_vector()

    def __len__(self):
        return self._length

    def _reset_vector(self):
        # If default seed (= None) is used, random vector will be generated
        np.random.seed(self._seed)

        vec = [np.random.random((1, self._dim)) for _ in range(self._length)]
        self._iter = iter(vec)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            vector = next(self._iter)
        except StopIteration:
            self._reset_vector()
            vector = next(self._iter)

        return torch.tensor(vector, dtype=torch.float64)
