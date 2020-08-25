#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time

from torch.utils.data import DataLoader
from torch.optim import Adam

from config import load_config
from common import write_json
from dataset import DCGanDataset
from model import DCGanGenerator, DCGanDiscriminator, load_pretrained_model
from trainer import DCGanTrainer, ImageEvaluator


# Logging
from logging import getLogger, StreamHandler, INFO
logger = getLogger()
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='config/default.json',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', '--output', '--out', '-o',
                        default='result',
                        help='Path to output directory')
    parser.add_argument('--gpu', '-g', type=int, default=None,
                        help='GPU id (default is cpu)')
    parser.add_argument('--label', '-l', required=True,
                        help='Training dataset label')
    parser.add_argument('--max_epoch', '-m', type=int, default=-1,
                        help='When the epoch reach this value, stop training,')
    args = parser.parse_args()
    return args


def setup_dataset(config, label):
    train_dir = os.path.join(config['dataset']['train']['dirname'], label)
    test_dir = os.path.join(config['dataset']['test']['dirname'], label)
    assert os.path.exists(train_dir)
    assert os.path.exists(test_dir)

    img_size = 4 * 2 ** config['model']['depth']

    train_dataset = DCGanDataset(train_dir, train=True,
                                 img_size=(img_size, img_size),
                                 ext=config['dataset']['train']['ext'],
                                 dimension=config['model']['in_ch'])
    test_dataset = DCGanDataset(test_dir, img_size=(img_size, img_size),
                                ext=config['dataset']['test']['ext'],
                                dimension=config['model']['in_ch'],
                                seed=config['seed'])

    train_loader = DataLoader(train_dataset, **config['loader'])
    test_loader = DataLoader(test_dataset, **config['loader'])

    return dict(train_loader=train_loader, test_loader=test_loader)


def setup_model(config):
    in_ch, out_ch = config['model']['in_ch'], config['model']['out_ch']
    depth = config['model']['depth']
    detach = config['model']['detach']

    generator = DCGanGenerator(in_ch, out_ch, depth=depth, detach=detach)
    discriminator = DCGanDiscriminator(out_ch, depth=depth, detach=detach)

    if config['model']['pretrained']['generator'] is None:
        load_pretrained_model(generator,
                              config['model']['pretrained']['generator'])
    if config['model']['pretrained']['discriminator'] is None:
        load_pretrained_model(discriminator,
                              config['model']['pretrained']['discriminator'])

    if config['optimizer']['generator'] is not None:
        gen_optimizer = Adam(generator.parameters(),
                             **config['optimizer']['generator'])
    else:
        gen_optimizer = None

    if config['optimizer']['discriminator'] is not None:
        dis_optimizer = Adam(discriminator.parameters(),
                             **config['optimizer']['discriminator'])
    else:
        dis_optimizer = None

    return {
        'generator': generator,
        'discriminator': discriminator,
        'gen_optimizer': gen_optimizer,
        'dis_optimizer': dis_optimizer,
    }


def setup_trainer(config, save_dir, device, datasets, models):
    evaluator = ImageEvaluator(save_dir,
                               config['evaluator_interval']['image'])
    snapshot_interval = config['evaluator_interval']['snapshot']
    trainer = DCGanTrainer(save_dir, **datasets, **models, device=device,
                           evaluator=evaluator,
                           snapshot_interval=snapshot_interval)

    return trainer


def main(argv):
    args = parse_arguments(argv)

    config = load_config(args.config)
    datasets = setup_dataset(config, args.label)
    models = setup_model(config)

    ctime = time.strftime('%y%m%d_%H%M')
    dst_dir = os.path.join(args.output_dir, '%s_%s' % (ctime, args.label))
    os.makedirs(dst_dir, exist_ok=True)

    write_json(os.path.join(dst_dir, 'config.json'), config)

    trainer = setup_trainer(config, dst_dir, args.gpu, datasets, models)
    trainer.run(config['iteraion_per_epoch'], args.max_epoch)


if __name__ == '__main__':
    main(sys.argv[1:])
