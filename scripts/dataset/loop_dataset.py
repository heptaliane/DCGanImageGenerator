# -*- coding: utf-8 -*-
import random
from queue import Empty
from multiprocessing import Queue

from torch.utils.data import IterableDataset


class LoopDataset(IterableDataset):
    def __init__(self, directory, loader, transform=None, shuffle=True):
        self._directory = directory
        self._loader = loader
        self.transform = transform
        self._names = directory.names
        self._shuffle = shuffle
        self._queue = Queue()
        self._reset_queue()

    def _reset_queue(self):
        if self._shuffle:
            random.shuffle(self._names)
        for name in self._names:
            self._queue.put(name)

    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            name = self._queue.get(timeout=0.1)
        except Empty:
            self._reset_queue()
            name = self._queue.get()

        path = self._directory.name_to_path(name)
        data = self._loader(path)
        if self.transform is not None:
            data = self.transform(data)

        return data
