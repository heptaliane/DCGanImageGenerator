#!/usr/bin/env python3
# -*- coding; utf-8 -*-
import os
import sys
import argparse

import cv2
from urllib import request

from common import get_filenames, read_json, write_json

# Logging
from logging import getLogger, StreamHandler, INFO
logger = getLogger()
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '--input', '-i', required=True,
                        help='Path to input image directory')
    parser.add_argument('--output_dir', '--out', '-o',
                        default='data/extracted',
                        help='Path to output image directory')
    parser.add_argument('--img_size', '-s', type=int, default=200,
                        help='Extracted image size (defualt = 200 px)')
    parser.add_argument('--detector_path', default='lbpcascade_animeface.xml',
                        help='Path to detector xml')
    args = parser.parse_args(argv)

    return args


class AnimeFaceDetector():
    _CASCADE_URL = 'https://raw.githubusercontent.com/nagadomi/' + \
                   'lbpcascade_animeface/master/lbpcascade_animeface.xml'

    def __init__(self, cascade_path, min_size=100):
        if not os.path.exists(cascade_path):
            logger.info('Cascade file does not exist: Downloading...')
            request.urlretrieve(self._CASCADE_URL, cascade_path)

        self._detector = cv2.CascadeClassifier(cascade_path)
        self._min_size = (min_size, min_size)

    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        return self._detector.detectMultiScale(gray, minSize=self._min_size)


class AnimeFaceExtractor():
    _CACHE_FILENAME = '.rect.json'

    def __init__(self, cascade_path, dst_dir,
                 image_size=200, margin=0.2, max_scale=1.2):
        min_size = int(image_size / (1.0 + margin) / max_scale)
        self._detector = AnimeFaceDetector(cascade_path, min_size=min_size)
        self.dst_dir = dst_dir
        os.makedirs(self.dst_dir, exist_ok=True)
        self._image_size = (image_size, image_size)
        self._margin = margin
        rects = read_json(os.path.join(dst_dir, self._CACHE_FILENAME))
        self._rects = dict() if rects is None else rects

    def _save_extract_face(self, extracted, filename):
        extracted = cv2.resize(extracted, self._image_size)
        dstpath = os.path.join(self.dst_dir, filename)
        cv2.imwrite(dstpath, extracted)

    def __call__(self, src_path):
        basename = os.path.splitext(os.path.basename(src_path))[0]
        filename = '%s_%s.jpg' % (basename, '%02d')
        if basename in self._rects:
            logger.info('"%s" exists... Skip', basename)
            return

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        rects = self._detector(img)
        self._rects[basename] = list()

        src_h, src_w = img.shape[:2]
        for i, (x, y, w, h) in enumerate(rects):
            cnt_x, cnt_y = x + w // 2, y + h // 2
            cnt_d = int(max(w, h) * (1.0 + self._margin) * 0.5)
            x0, y0 = max(cnt_x - cnt_d, 0), max(cnt_y - cnt_d, 0)
            x1, y1 = min(cnt_x + cnt_d, src_w), min(cnt_y + cnt_d, src_h)

            if x1 - x0 < cnt_d * 2:
                if x0 > 0:
                    x0 = max(x1 - cnt_d * 2, 0)
                elif x1 < src_w:
                    x1 = min(x0 + cnt_d * 2, src_w)
            if y1 - y0 < cnt_d * 2:
                if y0 > 0:
                    y0 = max(y1 - cnt_d * 2, 0)
                elif y1 < src_h:
                    y1 = min(y0 + cnt_d * 2, src_h)

            if x1 - x0 < cnt_d * 2 and x1 - x0 < y1 - y0:
                l = x1 - x0
                y0 = cnt_y - l // 2
                y1 = y0 + l
            elif y1 - y0 < cnt_d * 2 and x1 - x0 > y1 - y0:
                l = y1 - y0
                x0 = cnt_x - l // 2
                x1 = x0 + l

            self._save_extract_face(img[y0:y1, x0:x1, :],
                                    filename % i)
            self._rects[basename].append([int(x0), int(x1), int(y0), int(y1)])
        write_json(os.path.join(self.dst_dir, self._CACHE_FILENAME),
                   self._rects)


def main(argv):
    args = parse_arguments(argv)

    dirname = os.path.basename(args.input_dir)
    output_dir = os.path.join(args.output_dir, dirname)

    extractor = AnimeFaceExtractor(args.detector_path, output_dir,
                                   image_size=args.img_size)

    names = get_filenames(args.input_dir)
    for name in names[::-1]:
        logger.info('Extract face from "%s"', name)
        extractor(os.path.join(args.input_dir, name))


if __name__ == '__main__':
    main(sys.argv[1:])
