"""
Test case for util module.
"""

import logging
import os
import time
import unittest

import pandas as pd

from billys.util import (BILLYS_WORKSPACE_NAME, get_data_home, get_data_tmp, identity, get_filename,
                         make_dataset_filename, ensure_dir, now, get_elapsed_time, get_log_level)

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class UtilTest(unittest.TestCase):

    def test_indentity(self):
        self.assertEqual(1, identity(1))
        self.assertEqual(None, identity(None))
        self.assertEqual(identity, identity(identity))

    def test_get_data_home(self):
        # get data home with default value
        self.assertEqual(os.path.join(os.path.expanduser('~'), '.billys'),
                         get_data_home())

        # get data home with given path
        self.assertEqual('/path/to/billys/home',
                         get_data_home('/path/to/billys/home'))

    def test_get_data_tmp(self):
        # get data home with default value
        self.assertEqual(os.path.join(os.path.expanduser('~'), '.billys', '.tmp'),
                         get_data_tmp())

        # get data home with given path
        self.assertEqual('/path/to/billys/temp',
                         get_data_tmp('/path/to/billys/temp'))

    def test_get_filename(self):
        self.assertEqual('/path/to/data/foo.ext',
                         get_filename(name='foo.ext', data_home='/path/to/data'))

    def test_get_make_dataset_filename(self):
        self.assertEqual(make_dataset_filename(filename='foo.ext', cat='bar', subset='baz', step='zoo',
                                               data_home='path/to/datahome'), 'path/to/datahome/zoo/baz/bar/foo.jpg')

    def test_ensure_dir(self):
        ensure_dir('/tmp/path/to/mydata/file.ext')
        self.assertTrue(os.path.exists('/tmp/path/to/mydata'))

    def test_get_esapsed_time(self):
        self.assertEqual(0, get_elapsed_time(5, 5))
        self.assertEqual(2, get_elapsed_time(5, 7))

    def test_get_log_level(self):
        self.assertEqual(logging.NOTSET, get_log_level('foo'))
        self.assertEqual(logging.DEBUG, get_log_level('debug'))
        self.assertEqual(logging.INFO, get_log_level('info'))
        self.assertEqual(logging.WARNING, get_log_level('warning'))
        self.assertEqual(logging.WARN, get_log_level('warn'))
        self.assertEqual(logging.ERROR, get_log_level('error'))
        self.assertEqual(logging.CRITICAL, get_log_level('critical'))
        self.assertEqual(logging.FATAL, get_log_level('fatal'))
        self.assertEqual(logging.DEBUG, get_log_level('  DeBug  '))
