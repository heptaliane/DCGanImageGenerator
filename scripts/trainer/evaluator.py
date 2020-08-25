# -*- coding: utf-8 -*-
import os

import math

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToPILImage

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class BestModelWriter():
    def __init__(self, save_dir, name=None):
        name = 'model_best.pth' if name is None else '%s.pth' % name
        self._dst_path = os.path.join(save_dir, name)
        self._loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def load_state_dict(self, state):
        self._loss = state['loss']

    def state_dict(self):
        return dict(self._loss)

    def update(self, loss, model):
        if self._loss > loss:
            self._loss = loss
            torch.save(model.state_dict(), self._dst_path)
            logger.info('Save best model (%s).', self._dst_path)


class SnapshotWriter():
    def __init__(self, save_dir):
        self._dst_path = os.path.join(save_dir, 'snapshot_latest.pth')
        os.makedirs(save_dir, exist_ok=True)

    def update(self, trainer):
        torch.save(trainer.state_dict(), self._dst_path)


class ImageEvaluator():
    def __init__(self, save_dir, interval=0):
        self.save_dir = save_dir
        self.interval = interval
        self.tensor_to_image = ToPILImage()

    def _create_thumbnail(self, batches, dst_path):
        n, c, h, w = batches.size()
        n_rows = math.ceil(math.sqrt(n))
        n_cols = math.ceil(n / n_rows)
        thumb = np.zeros((h * n_rows, w * n_cols, c), dtype=np.uint8)

        for i in range(n):
            img = np.asarray(self.tensor_to_image(batches[i]))
            nx, ny = i % n_cols, math.floor(i / n_rows)
            thumb[ny * h: (ny + 1) * h, nx * w: (nx + 1) * w, :] = img

        thumb = Image.fromarray(thumb)
        thumb.save(dst_path)

    def __call__(self, preds, epoch):
        if self.interval <= 0:
            dst_dir = os.path.join(self.save_dir, 'test_result_latest')
        elif epoch % self.interval == 0:
            dst_dir = os.path.join(self.save_dir,
                                   'test_result_epoch%04d' % epoch)
        else:
            return
        os.makedirs(dst_dir, exist_ok=True)

        for i, pred in enumerate(preds):
            filename = 'test_result_idx_%03d.jpg' % i
            self._create_thumbnail(pred, os.path.join(dst_dir, filename))
