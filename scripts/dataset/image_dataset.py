# -*- codoing: utf-8 -*-
from torchvision import transforms
from PIL import Image

from .common import DatasetDirectory
from .loop_dataset import LoopDataset


def create_image_dataset(src_dir, train=False, img_size=(200, 200), ext='.jpg'):
    directory = DatasetDirectory(src_dir, ext)

    transform = [transforms.Resize(img_size)]
    if train:
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomRotation(15.0, expand=True))
        transform.append(transforms.RandomResizedCrop(img_size,
                                                      scale=(0.7, 1.0),
                                                      ratio=(1.0, 1.0)))
    transform.append(transforms.ToTensor())

    transform = transforms.Compose(transform)

    return LoopDataset(directory, Image.open, transform, train)
