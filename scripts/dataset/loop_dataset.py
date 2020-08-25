# -*- coding: utf-8 -*-
import random

from torch.utils.data import IterableDataset


class LoopDataset(IterableDataset):
    def __init__(self, directory, loader, transform=None, shuffle=True):
        self._directory = directory
        self._loader = loader
        self.transform = transform
        self._names = directory.names
        self._shuffle = shuffle
        self._reset_iter()

    def _reset_iter(self):
        if self._shuffle:
            random.shuffle(self._names)
        self._iter = iter(self._names)

    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            name = next(self._iter)
        except StopIteration:
            self._reset_iter()
            name = next(self._iter)

        path = self._directory.name_to_path(name)
        data = self._loader(path)
        if self.transform is not None:
            data = self.transform(data)

        return data
