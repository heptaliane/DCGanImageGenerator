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
                        help='Path to output base directory')
    parser.add_argument('--save_as', default=None,
                        help='Save images as a given label')
    parser.add_argument('--keyword', '--key', '-k', default=None,
                        help='Search image keyword')
    parser.add_argument('--user_id', '--user', '-u', type=int, default=None,
                        help='Search by given user id')
    parser.add_argument('--auth_json', '--auth', '-a',
                        default='.auth.json',
                        help='Path to pixiv authenticatation file')
    parser.add_argument('--allow_r18', action='store_true',
                        help='Allow R18 images')
    parser.add_argument('--allow_manga', action='store_true',
                        help='Allow manga images')
    parser.add_argument('--threshold_favorite', '-t', type=int, default=1000,
                        help='Threshold of the number of favorite')
    args = parser.parse_args()

    if args.keyword is None and args.user_id is None:
        logger.error('"--keyword" or "--user" argument is required.')
        args.print_help()
        exit()

    return args


class PixivDownloader():
    _NAME_FORMAT = 'pixiv_%s_%03d.jpg'

    def __init__(self, dst_dir, th_favorite=1000,
                 allow_r18=False, allow_manga=False):
        self._app = pixivpy3.AppPixivAPI()
        self._api = pixivpy3.PixivAPI()
        self._is_auth = False
        self._dst_dir = dst_dir
        self.th_favorite = th_favorite
        self.allow_manga = allow_manga
        self.allow_r18 = allow_r18
        os.makedirs(dst_dir, exist_ok=True)

    def authenticate(self, username, password):
        self._app.login(username, password)
        self._api.set_auth(self._app.access_token, self._app.refresh_token)
        self._is_auth = True

    def _download_images(self, illust_id):
        if os.path.exists(os.path.join(self._dst_dir,
                                       self._NAME_FORMAT % (illust_id, 0))):
            logger.info('"%s" exists... Skip', illust_id)
            return

        logger.info('Download "%s"..', illust_id)
        illusts = self._app.illust_detail(illust_id).illust
        if illusts is None:
            logger.error('Cannot load image (%d)' % illust_id)
        if illusts.page_count == 1:
            url = illusts.meta_single_page.original_image_url
            self._app.download(url, name=self._NAME_FORMAT % (illust_id, 0),
                               path=self._dst_dir)
        else:
            for i, page in enumerate(illusts.meta_pages):
                url = page.image_urls.original
                self._app.download(url,
                                   name=self._NAME_FORMAT % (illust_id, i),
                                   path=self._dst_dir)

    def _filter_illusts(self, res, keyword):
        if res.stats.scored_count < self.th_favorite:
            return False
        if not self.allow_r18 and res.sanity_level != 'white':
            return False
        if not self.allow_manga and res.type == 'manga':
            return False
        return keyword is None or keyword in res.tags

    def __call__(self, keyword=None, user_id=None):
        if not self._is_auth:
            logger.info('Not authenticated.')
            return

        for i in range(1, 1000):
            if user_id is not None:
                data = self._api.users_works(user_id, page=i)
            elif keyword is not None:
                data = self._api.search_works(keyword, mode='tag', page=i)

            if data.status != 'success':
                break

            for res in data.response:
                if self._filter_illusts(res, keyword):
                    self._download_images(res.id)


def main(argv):
    args = parse_arguments(argv)

    if args.save_as is None:
        label = str(args.user_id) if args.keyword is None \
            else base64_encode(args.keyword)
        out_dir = os.path.join(args.out_dir, label)
    else:
        out_dir = os.path.join(args.out_dir, base64_encode(args.save_as))
    downloader = PixivDownloader(out_dir, args.threshold_favorite,
                                 args.allow_r18, args.allow_manga)

    auth = read_json(args.auth_json)
    auth = dict() if auth is None else auth
    while True:
        if 'pixiv' not in auth:
            auth['pixiv'] = {
                'username': input('Username >> '),
                'password': getpass('Password >> '),
            }

        try:
            downloader.authenticate(**auth['pixiv'])
            write_json(args.auth_json, auth)
            break
        except pixivpy3.PixivError:
            logger.error('Failed to login.')
            del auth['pixiv']

    downloader(keyword=args.keyword, user_id=args.user_id)


if __name__ == '__main__':
    main(sys.argv[1:])
