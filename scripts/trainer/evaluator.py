# -*- coding: utf-8 -*-
import os

import torch

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
        self._dst_path = os.path.join(self.save_dir, 'snapshot_latest.pth')
        os.makedirs(save_dir, exist_ok=True)

    def update(self, trainer):
        torch.save(trainer.state_dict(), self._dst_path)
