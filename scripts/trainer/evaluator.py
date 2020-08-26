# -*- coding: utf-8 -*-
import os
from collections import deque

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
        return dict(loss=self._loss)

    def update(self, loss, model):
        if self._loss > loss:
            self._loss = loss
            torch.save(model.state_dict(), self._dst_path)
            logger.info('Save best model (%s).', self._dst_path)


class LocalBestModelWriter():
    def __init__(self, save_dir, name=None, epochs=10):
        name = 'model_best.pth' if name is None \
                else '%s.pth' % name
        self._dst_path = os.path.join(save_dir, name)
        self._loss = deque([float('inf')], epochs)
        os.makedirs(save_dir, exist_ok=True)

    def load_state_dict(self, state):
        self._loss = deque(state['loss'], state['epochs'])

    def state_dict(self):
        return {
            'loss': list(self._loss),
            'epochs': len(self._loss),
        }

    def update(self, loss, model):
        if loss < min(self._loss):
            torch.save(model.state_dict(), self._dst_path)
            logger.info('Save local best model (%s).', self._dst_path)
        self._loss.append(loss)


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

    def _normalize_input(self, tensor):
        tmin, tmax = float(tensor.min()), float(tensor.max())
        tensor.clamp_(min=tmin, max=tmax)
        return tensor.add_(-tmin).div(tmax - tmin + 1e-5)

    def _create_thumbnail(self, batches, dst_path):
        n, c, h, w = batches.size()
        n_rows = math.ceil(math.sqrt(n))
        n_cols = math.ceil(n / n_rows)
        thumb = np.zeros((h * n_rows, w * n_cols, c), dtype=np.uint8)

        for i in range(n):
            pred = self._normalize_input(batches[i])
            img = np.asarray(self.tensor_to_image(pred))
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
            filename = 'test_result_idx_%02d.jpg' % i
            self._create_thumbnail(pred, os.path.join(dst_dir, filename))


class DCGanEvaluator(ImageEvaluator):
    def __init__(self, save_dir, interval=0):
        super().__init__(save_dir, interval)
        self.save_dir = os.path.join(self.save_dir, 'test_result')
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, preds, epoch):
        if self.interval > 0 and epoch % self.interval > 0:
            return

        batches = list()
        for pred in preds:
            for i in range(len(pred)):
                batches.append(pred[i])

        filename = 'test_result_epoch_%04d.jpg' % epoch
        dst_path = os.path.join(self.save_dir, filename)
        self._create_thumbnail(torch.stack(batches, dim=0), dst_path)
