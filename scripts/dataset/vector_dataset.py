# -*- coding: utf-8 -*-
from queue import Empty
from multiprocessing import Queue

import numpy as np

import torch
from torch.utils.data import IterableDataset


class VectorDataset(IterableDataset):
    def __init__(self, dimension, length, seed=None):
        self._length = length
        self._dim = dimension
        self._seed = seed
        self._queue = Queue()
        self._reset_vector()

    def __len__(self):
        return self._length

    def _reset_vector(self):
        # If default seed (= None) is used, random vector will be generated
        np.random.seed(self._seed)

        for _ in range(self._length):
            self._queue.put(np.random.random((self._dim, 1, 1)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            vector = self._queue.get(timeout=0.1)
        except Empty:
            self._reset_vector()
            vector = self._queue.get()

        return torch.tensor(vector, dtype=torch.float32)
