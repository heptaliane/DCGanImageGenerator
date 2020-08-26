# -*- codoing: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .common import DatasetDirectory


class ImageDataset(Dataset):
    def __init__(self, directory, transform):
        self._directory = directory
        self._transform = transform
        self._names = directory.names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        name = self._names[idx]
        path = self._directory.name_to_path(name)
        img = Image.open(path)

        return self._transform(img)


def create_image_dataset(src_dir, train=False,
                         img_size=(200, 200), ext='.jpg'):
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

    return ImageDataset(directory, transform)
