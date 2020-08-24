# -*- coding: utf-8 -*-
from torch.utils.data import IterableDataset

from .image_dataset import create_image_dataset
from .vector_dataset import VectorDataset


class DCGanDataset(IterableDataset):
    def __init__(self, src_dir, train=False, img_size=(200, 200), ext='.jpg',
                 dimension=100, seed=0):
        self.dis_dataset = create_image_dataset(src_dir, train=train,
                                                img_size=img_size, ext=ext)
        length = len(self.dis_dataset)
        self.gen_dataset = VectorDataset(dimension, length,
                                         seed=None if train else 0)

    def __len__(self):
        return len(self.dis_dataset)

    def __iter__(self):
        return self

    def __next__(self):
        return {
            'generator': next(self.gen_dataset),
            'discriminator': next(self.dis_dataset),
        }
