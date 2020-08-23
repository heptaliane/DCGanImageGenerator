#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from getpass import getpass

from urllib import request
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

from common import read_json, write_json, base64_encode

# Logging
from logging import getLogger, StreamHandler, INFO, WARNING
from selenium.webdriver.remote import remote_connection
logger = getLogger()
logger.setLevel(INFO)
logger.addHandler(StreamHandler())
remote_connection.LOGGER.setLevel(WARNING)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '--out', '-o', default='data/orginal',
                        help='Path to output base directory')
    parser.add_argument('--save_as', default=None,
                        help='Save images as a given label')
    parser.add_argument('--keyword', '--key', '-k', required=True,
                        help='Search image keyword')
    parser.add_argument('--auth_json', '--auth', '-a',
                        default='.auth.json',
                        help='Path to niconico authenticatation file')
    args = parser.parse_args()

    return args


class NiconicoDownloader():
    _NAME_FORMAT = 'nico_%s.jpg'

    def __init__(self, dst_dir):
        self._dst_dir = dst_dir
        self._driver = None
        self._is_auth = False

    def set_firefox_driver(self, headless=True, private=True):
        if self._driver is not None:
            self._driver.quit()

        options = webdriver.FirefoxOptions()
        profile = webdriver.FirefoxProfile()
        if headless:
            options.add_argument('-headless')
        if private:
            profile.set_preference('browser.privatebrowsing.autostart', True)
        self._driver = webdriver.Firefox(firefox_profile=profile,
                                         options=options)

    def authenticate(self, username, password):
        if self._driver is None:
            self.set_firefox_driver()

        self._driver.get('https://account.nicovideo.jp/login')
        user_input = self._driver.find_element_by_id('input__mailtel')
        pass_input = self._driver.find_element_by_id('input__password')
        submit = self._driver.find_element_by_id('login__submit')

        user_input.send_keys(username)
        pass_input.send_keys(password)
        submit.click()

        self._driver.implicitly_wait(1.0)
        name = self._driver.find_element_by_id('CommonHeader').\
            find_elements_by_tag_name('span')[0].text
        logger.info('Login as "%s".', name)
        self._is_auth = True

    def _download_image(self, illust_id):
        dst_path = os.path.join(self._dst_dir, self._NAME_FORMAT % illust_id)
        if os.path.exists(dst_path):
            logger.info('"%s" exists... Skip', illust_id)
            return
        logger.info('Download "%s"..', illust_id)

        viewer_url = 'https://seiga.nicovideo.jp/image/source/%s' % illust_id
        self._driver.get(viewer_url)
        url = self._driver.find_elements_by_tag_name('img')[-1].\
            get_attribute('src')
        try:
            with request.urlopen(url) as img:
                with open(dst_path, 'wb') as f:
                    f.write(img.read())
        except IOError:
            logger.error('"%s" is not found', url)
        except UnicodeEncodeError:
            logger.error('Invalid format ("%s")', url)

    def _search_illusts(self, search_word, page):
        url = 'https://seiga.nicovideo.jp/tag/%s?page=%d' % (search_word, page)
        self._driver.get(url)

        elems = self._driver.find_elements_by_tag_name('li')
        urls = [elem.find_elements_by_tag_name('a')[0].get_attribute('href')
                for elem in elems
                if elem.get_attribute('class').startswith('list_item')]
        return [src.split('/')[-1][2:] for src in urls]

    def __call__(self, search_word):
        if not self._is_auth:
            logger.error('Not authenticated.')
            return

        i = 1
        while True:
            illusts = self._search_illusts(search_word, i)
            if len(illusts) == 0:
                break

            for illust_id in illusts:
                self._download_image(illust_id)
            i += 1

    def __del__(self):
        self._driver.quit()


def main(argv):
    args = parse_arguments(argv)

    if args.save_as is None:
        out_dir = os.path.join(args.out_dir, base64_encode(args.keyword))
    else:
        out_dir = os.path.join(args.out_dir, base64_encode(args.save_as))
    downloader = NiconicoDownloader(out_dir)

    auth = read_json(args.auth_json)
    auth = dict() if auth is None else auth
    while True:
        if 'niconico' not in auth:
            auth['niconico'] = {
                'username': input('Username >> '),
                'password': getpass('Password >> '),
            }

        try:
            downloader.authenticate(**auth['niconico'])
            write_json(args.auth_json, auth)
            break
        except NoSuchElementException:
            logger.error('Failed to login.')
            del auth['niconico']

    downloader(args.keyword)


if __name__ == '__main__':
    main(sys.argv[1:])
