# ^*- coding: utf-8 -*-
from tqdm import tqdm

import torch
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter

from .evaluator import LocalBestModelWriter, SnapshotWriter,\
    PeriodicModelWriter
from .common import LoopIterator

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class DCGanTrainer():
    _STATE_KEYS = ('train_iter', 'test_iter', 'generator', 'discriminator',
                   'gen_model_writer', 'dis_model_writer', 'epoch')

    def __init__(self, save_dir, train_loader, test_loader,
                 generator, discriminator,
                 gen_optimizer=None, dis_optimizer=None,
                 device=None, evaluator=None,
                 snapshot_interval=10, model_save_interval=-1):
        self.device = torch.device('cpu') if device is None else device
        self.train_iter = LoopIterator(train_loader)
        self.test_iter = LoopIterator(test_loader)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.dis_loss = BCELoss()
        self.evaluator = evaluator
        self.batch_size = train_loader.batch_size
        self.snapshot_interval = snapshot_interval
        self.epoch = 0

        # Setup writer
        gen_name = 'best_generator'
        dis_name = 'best_discriminator'
        self.gen_model_writer = LocalBestModelWriter(save_dir, gen_name)
        self.dis_model_writer = LocalBestModelWriter(save_dir, dis_name)
        self.periodic_gen_writer = PeriodicModelWriter(save_dir, 'G',
                                                       model_save_interval)
        self.periodic_dis_writer = PeriodicModelWriter(save_dir, 'D',
                                                       model_save_interval)
        self.snapshot_writer = SnapshotWriter(save_dir)
        self.logger = SummaryWriter(save_dir)

        # Setup loss keys
        self._loss_keys = list()
        if self.gen_optimizer is not None:
            self._loss_keys.append('gen_loss')
        if self.dis_optimizer is not None:
            self._loss_keys.extend(['dis_loss',
                                    'dis_judge_real', 'dis_judge_fake'])

    def state_dict(self):
        state = dict()
        for key in self._STATE_KEYS:
            obj = getattr(self, key)
            if hasattr(obj, 'state_dict'):
                state[key] = obj.state_dict()
            else:
                state[key] = obj

    def load_state_dict(self, state):
        for lbl in state.keys():
            obj = getattr(self, lbl)
            if hasattr(obj, 'load_state_dict'):
                obj.load_state_dict(state[lbl])
            else:
                setattr(self, lbl, state[lbl])

    def _forward(self, inp, train=True):
        inp = {k: data.to(device=self.device) for k, data in inp.items()}
        loss = dict()

        if self.dis_optimizer is not None:
            # Forward discriminator
            self.discriminator.zero_grad()
            judge_real = self.discriminator.forward(inp['discriminator'])
            real_label = torch.full_like(judge_real, 1.0,
                                         dtype=torch.float32,
                                         device=self.device)
            dis_real_loss = self.dis_loss(judge_real, real_label)
            loss['dis_judge_real'] = judge_real.mean().item()

            # Backward
            if train:
                dis_real_loss.backward()

            # Create fake input
            fake = self.generator.forward(inp['generator'])

            # Forward discriminator
            judge_fake = self.discriminator.forward(fake.detach())
            fake_label = torch.full_like(judge_fake, 0.0,
                                         dtype=torch.float32,
                                         device=self.device)
            dis_fake_loss = self.dis_loss(judge_fake, fake_label)
            loss['dis_judge_fake'] = judge_fake.mean().item()
            loss['dis_loss'] = dis_fake_loss.item() + dis_real_loss.item()

            # Backward
            if train:
                dis_fake_loss.backward()
                self.dis_optimizer.step()

        if self.gen_optimizer is not None:
            # Forward generator
            self.generator.zero_grad()
            if self.dis_optimizer is None:
                fake = self.generator.forward(inp['generator'])
            judge = self.discriminator.forward(fake)
            real_label = torch.full_like(judge, 1.0,
                                         dtype=torch.float32,
                                         device=self.device)
            gen_loss = self.dis_loss(judge, real_label)
            loss['gen_loss'] = gen_loss.mean().item()

            # Backward
            if train:
                gen_loss.backward()
                self.gen_optimizer.step()

        if train:
            return loss
        else:
            return loss, fake.detach().cpu()

    def _train_step(self, n_train):
        self.generator.train()
        self.discriminator.train()

        avg_loss = {k: 0 for k in self._loss_keys}
        for _ in tqdm(range(n_train // self.batch_size)):
            data = next(self.train_iter)
            loss = self._forward(data, train=True)
            for k, v in loss.items():
                avg_loss[k] += v

        for k in avg_loss.keys():
            avg_loss[k] = avg_loss[k] / n_train * self.batch_size
            self.logger.add_scalar('train_%s' % k, avg_loss[k], self.epoch)
            logger.info('train_%s: %f', k, avg_loss[k])

    def _test_step(self):
        self.generator.eval()
        self.discriminator.eval()

        preds = list()
        avg_loss = {k: 0 for k in self._loss_keys}
        n_test = len(self.test_iter)
        for _ in tqdm(range(n_test)):
            data = next(self.test_iter)
            loss, pred = self._forward(data, train=False)
            preds.append(pred)
            for k, v in loss.items():
                avg_loss[k] += v

        for k in avg_loss.keys():
            avg_loss[k] = avg_loss[k] / n_test
            self.logger.add_scalar('test_%s' % k, avg_loss[k], self.epoch)
            logger.info('test_%s: %f', k, avg_loss[k])

        if self.gen_optimizer is not None:
            self.gen_model_writer.update(avg_loss['gen_loss'],
                                         self.generator)
            self.periodic_gen_writer.update(self.epoch, self.generator)
            self.evaluator(preds, self.epoch)
        if self.dis_optimizer is not None:
            self.dis_model_writer.update(avg_loss['dis_loss'],
                                         self.discriminator)
            self.periodic_dis_writer.update(self.epoch, self.discriminator)
        if self.epoch % self.snapshot_interval == 0:
            self.snapshot_writer.update(self)

    def run(self, n_train, max_epoch=-1):
        while True:
            self.epoch += 1
            logger.info('Epoch: %d', self.epoch)
            self._train_step(n_train)
            self._test_step()

            if 0 < max_epoch < self.epoch:
                logger.info('Reached max epoch')
                break
