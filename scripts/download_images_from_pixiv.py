#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from getpass import getpass

import pixivpy3

from common import base64_encode, read_json, write_json

# Logging
from logging import getLogger, StreamHandler, INFO
logger = getLogger()
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '--out', '-o', default='data/orginal',
                        help='Path to output directory')
    parser.add_argument('--keyword', '--key', '-k', required=True,
                        help='Search image keyword')
    parser.add_argument('--auth_json', '--auth', '-a',
                        default='.auth.json',
                        help='Path to pixiv authenticatation file')
    args = parser.parse_args()

    return args


class PixivDownloader():
    def __init__(self, dst_dir):
        self._app = pixivpy3.AppPixivAPI()
        self._api = pixivpy3.PixivAPI()
        self._is_auth = False
        self._dst_dir = dst_dir
        os.makedirs(dst_dir, exist_ok=True)

    def authenticate(self, username, password):
        self._app.login(username, password)
        self._api.set_auth(self._app.access_token, self._app.refresh_token)
        self._is_auth = True

    def _download_images(self, illust_id):
        if os.path.exists(os.path.join(self._dst_dir, '%s_000.jpg' % illust_id)):
            logger.info('"%s" exists... Skip', illust_id)
            return

        logger.info('Download "%s"..', illust_id)
        illusts = self._app.illust_detail(illust_id).illust
        if illusts.page_count == 1:
            url = illusts.meta_single_page.original_image_url
            self._app.download(url, '%s_000.jpg' % illust_id, path=self._dst_dir)
        else:
            for i, page in enumerate(illusts.meta_pages):
                url = page.image_urls.original
                self._app.download(url, name='%s_%03d.jpg' % (illust_id, i),
                                   path=self._dst_dir)

    def __call__(self, word):
        if not self._is_auth:
            logger.info('Not authenticated.')
            return

        for i in range(1, 1000):
            data = self._api.search_works(word, mode='tag', page=i)
            if data.status != 'success':
                break

            for res in data.response:
                self._download_images(res.id)


def main(argv):
    args = parse_arguments(argv)

    out_dir = os.path.join(args.out_dir, base64_encode(args.keyword))
    downloader = PixivDownloader(out_dir)

    auth = read_json(args.auth_json)
    while True:
        if auth is None:
            auth = {
                'username': input('Username >> '),
                'password': getpass('Password >> '),
            }

        try:
            downloader.authenticate(**auth)
            write_json(args.auth_json, auth)
            break
        except pixivpy3.PixivError:
            logger.error('Failed to login.')
            auth = None

    downloader(args.keyword)


if __name__ == '__main__':
    main(sys.argv[1:])
