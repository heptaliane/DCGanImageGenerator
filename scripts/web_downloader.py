# -*- coding: utf-8 -*-
import re
import os
from abc import ABCMeta, abstractmethod
from io import BytesIO

from urllib import request, parse
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException

# Logging
from logging import getLogger, NullHandler, WARNING
from selenium.webdriver.remote import remote_connection
logger = getLogger(__name__)
logger.addHandler(NullHandler())
remote_connection.LOGGER.setLevel(WARNING)


class WebDownloader(metaclass=ABCMeta):
    _ESCAPE_CHAR = re.compile(r'[^\w\'\(\)]+')
    _DOWNLOAD_HEADER = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:79.0)' +
                      ' Gecko/20100101 Firefox/79.0 '
    }

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._driver = None
        os.makedirs(self.save_dir, exist_ok=True)

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

    def _download_image(self, url, filename):
        filename = filename.replace('%', '=')
        req = request.Request(url=url, headers=self._DOWNLOAD_HEADER)
        with request.urlopen(req) as f:
            img = Image.open(BytesIO(f.read())).convert('RGBA')
            background = Image.new('RGBA', img.size, (255, 255, 255))
            img = Image.alpha_composite(background, img).convert('RGB')
            img.save(os.path.join(self.save_dir, filename), 'jpeg')

    def _get_label_from_url(self, url):
        return self._ESCAPE_CHAR.sub('_', parse.unquote(url.split('/')[-1]))

    def _download_image_from_page(self, url):
        label = self._get_label_from_url(url)
        filename = '%s_%s.jpg' % (label, '%d')
        if os.path.exists(os.path.join(self.save_dir, filename % 0)):
            logger.info('"%s" exists... skip', label)
            return

        self._driver.get(url)
        srcs = self._collect_image_sources()

        i = 0
        for src in srcs:
            target = self._parse_url_from_source(src)
            self._download_image(target, filename % i)
            logger.info('Download "%s".', filename % i)
            i += 1

    @abstractmethod
    def _collect_image_sources(self):
        raise NotImplementedError

    @abstractmethod
    def _collect_page_urls(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_url_from_source(self, url):
        raise NotImplementedError

    def __call__(self):
        if self._driver is None:
            self.set_firefox_driver()

        urls = self._collect_page_urls()
        errors = 0
        for url in urls:
            try:
                self._download_image_from_page(url)
            except StaleElementReferenceException:
                label = self._get_label_from_url(url)
                sample_path = os.path.join(self.save_dir, '%s_0.jp' % label)
                errors += 1
                if os.path.exists(sample_path):
                    os.remove(sample_path)

        if errors > 0:
            logger.error('Failed to download %d images', errors)
            logger.error('Please try again.')

    def __del__(self):
        self._driver.quit()
