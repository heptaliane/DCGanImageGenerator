# -*- coding: utf-8 -*-
import os
import sys
import argparse
import shutil
import enum

import cv2

from common import get_filenames, read_json, write_json


class KeyStatus(enum.IntEnum):
    QUIT = 113
    UNDO = 117
    OK = 106
    FAIL = 102
    REMOVE = 100


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '--input', '-i', required=True,
                        help='Path to input directory')
    parser.add_argument('--output_dir', '--output', '--out', '-o',
                        default='data/validate',
                        help='Path to output base directory')
    parser.add_argument('--scale', '-s', type=float, default=3.0,
                        help='Image preview scale')
    args = parser.parse_args()
    return args


def show_image(src_path, scale=1.0):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    size = (int(img.shape[0] * scale), int(img.shape[1] * scale))
    img = cv2.resize(img, size)

    cv2.imshow('preview', img)
    while True:
        key = cv2.waitKey(0)
        try:
            status = KeyStatus(key)
            return status
        except ValueError:
            pass


def main(argv):
    args = parse_arguments(argv)

    dirname = os.path.basename(args.input_dir)
    valid_dir = os.path.join(args.output_dir, dirname, 'valid')
    invalid_dir = os.path.join(args.output_dir, dirname, 'invalid')
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)
    removed_json = os.path.join(args.output_dir, dirname, '.cache.json')

    names = get_filenames(args.input_dir)
    valid_names = get_filenames(valid_dir)
    invalid_names = get_filenames(invalid_dir)
    removed_names = read_json(removed_json)
    removed_names = [] if removed_names is None else removed_names
    names = sorted(set(names) - set(valid_names) - set(invalid_names)
                   - set(removed_names))

    # Instruction
    sys.stdout.write(
        'Key input instructions:\n'
        'j: Accept current image\n'
        'k: Reject current image\n'
        'u: Undo recent validation\n'
        'd: Exclude image \n'
        'q: Quit validation\n'
    )

    i = 0
    while i < len(names):
        path = os.path.join(args.input_dir, names[i])
        key = show_image(path, args.scale)

        if key == KeyStatus.UNDO and i > 1:
            i -= 1
            if os.path.exists(os.path.join(valid_dir, names[i])):
                os.remove(os.path.join(valid_dir, names[i]))
            else:
                os.remove(os.path.join(invalid_dir, names[i]))
        elif key == KeyStatus.OK:
            shutil.copyfile(path, os.path.join(valid_dir, names[i]))
            i += 1
        elif key == KeyStatus.FAIL:
            shutil.copyfile(path, os.path.join(invalid_dir, names[i]))
            i += 1
        elif key == KeyStatus.REMOVE:
            removed_names.append(names[i])
            write_json(removed_json, removed_names)
            i += 1
        else:
            exit()


if __name__ == '__main__':
    main(sys.argv[1:])
