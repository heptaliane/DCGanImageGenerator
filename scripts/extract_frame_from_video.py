#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import argparse

import cv2

from extract_face_images import AnimeFaceExtractor

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '--out', '-o', default='data/extracted',
                        help='Path to output base directory')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to source video')
    parser.add_argument('--start_sec', '-s', type=float, default=0.0,
                        help='Extract start second')
    parser.add_argument('--end_sec', '-e', type=float, default=float('inf'),
                        help='Extract end second')
    parser.add_argument('--interval', '-I', type=float, default=1.0,
                        help='Extract interval second')
    parser.add_argument('--img_size', '--size', type=int, default=200,
                        help='Extracted image size (defualt = 200 px)')
    parser.add_argument('--detector_path', default='lbpcascade_animeface.xml',
                        help='Path to detector xml')
    args = parser.parse_args()

    return args


class VideoFrameExtractor():
    def __init__(self, save_dir, interval, extractor):
        self.save_dir = save_dir
        self.interval = interval
        self.extractor = extractor
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, path, start_sec=0.0, end_sec=float('inf')):
        video = cv2.VideoCapture(path)
        assert video.isOpened()

        basename = os.path.splitext(os.path.basename(path))[0]
        dst_dir = os.path.join(self.save_dir, basename)
        os.makedirs(dst_dir, exist_ok=True)
        self.extractor.dst_dir = dst_dir

        fps = video.get(cv2.CAP_PROP_FPS)

        i = 0
        sec = start_sec
        lbl = '%s_%s' % (basename, '%06d')
        while True:
            video.set(cv2.CAP_PROP_POS_FRAMES, round(sec * fps))
            ret, frame = video.read()

            if ret:
                self.extractor.extract_face(frame, lbl % i)
            else:
                break

            i += 1
            sec += self.interval
            if sec > end_sec:
                break


def main(argv):
    args = parse_arguments(argv)

    extractor = AnimeFaceExtractor(args.detector_path, args.out_dir,
                                   image_size=args.img_size)
    frame_extractor = VideoFrameExtractor(args.out_dir, args.interval,
                                          extractor)

    frame_extractor(args.input, args.start_sec, args.end_sec)


if __name__ == '__main__':
    main(sys.argv[1:])
