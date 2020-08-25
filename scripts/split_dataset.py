#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
import shutil

from common import get_filenames

# Logging
from logging import getLogger, StreamHandler, INFO
logger = getLogger()
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


def parse_argments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '--input', '-i', required=True,
                        help='Path to input directory')
    parser.add_argument('--output_dir', '--output', '-o',
                        default='data/',
                        help='Path to output base directory')
    parser.add_argument('--n_test', '-n', type=int, default=128,
                        help='The number of files for test')
    args = parser.parse_args()

    return args


def main(argv):
    args = parse_argments(argv)

    basename = args.input_dir.split('/')[-2]
    train_dir = os.path.join(args.output_dir, 'train', basename)
    test_dir = os.path.join(args.output_dir, 'test', basename)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    names = set(get_filenames(args.input_dir))
    test_names = set(get_filenames(test_dir))
    train_names = set(get_filenames(train_dir))
    invalid_names = test_names & train_names
    names -= test_names
    names -= train_names

    for name in invalid_names:
        os.remove(os.path.join(test_dir, name))
        logger.info('"%s" is duplicated... Remove', name)

    n_test = args.n_test - len(test_names)
    test_samples = random.sample(names, n_test)
    for name in names:
        src_path = os.path.join(args.input_dir, name)
        if name in test_samples:
            dst_path = os.path.join(test_dir, name)
        else:
            dst_path = os.path.join(train_dir, name)

        shutil.copyfile(src_path, dst_path)


if __name__ == '__main__':
    main(sys.argv[1:])
