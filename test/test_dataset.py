import logging
import os
import time
import unittest

import pandas as pd
from sklearn.utils import Bunch

from billys.dataset import fetch_billys, make_dataframe, is_good, is_pdf, is_valid

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class UtilTest(unittest.TestCase):

    def test_fetch_billys(self):
        # TODO
        pass

    def test_make_dataframe(self):
        dataset = Bunch(filenames=['/path/to/data/mydataset/train/cat1/1.jpg', '/path/to/data/mydataset/train/cat1/2.jpg',
                                   '/path/to/data/mydataset/train/cat2/1.jpg', '/path/to/data/mydataset/train/cat2/2.jpg'],
                        target=[0, 0, 1, 1],
                        target_names=['cat1', 'cat2'])
        df = make_dataframe(dataset=dataset, force_good=True, subset='train')

        # Check columns
        actual_cols = df.columns.to_list().copy()
        actual_cols.sort()
        expected_cols = ['filename', 'target', 'target_name',
                         'grayscale', 'is_good', 'is_pdf', 'subset']
        expected_cols.sort()
        self.assertEqual(expected_cols, actual_cols)

        # Check not empty
        self.assertTrue(len(df['filename'].tolist()) > 0)

        # Check subset
        self.assertTrue((df['subset'] == 'train').all())

        # Check good
        self.assertTrue((df['is_good'] == True).all())

        # Check target names
        target_names = df['target_name'].unique().tolist()
        target_names.sort()
        self.assertEqual(['cat1', 'cat2'], target_names)

    def test_is_good(self):
        self.assertTrue(is_good('/path/to/mydata/file.pdf'))
        self.assertTrue(is_good('/path/to/mydata/file.pdf', force=True))
        self.assertTrue(is_good('/path/to/mydata/file.jpg', force=True))
        self.assertFalse(is_good('/path/to/mydata/file.jpg'))
        self.assertFalse(is_good('/path/to/mydata/file'))

    def test_is_pdf(self):
        self.assertTrue(is_pdf('/path/to/mydata/file.pdf'))
        self.assertFalse(is_pdf('/path/to/mydata/file.jpg'))
        self.assertFalse(is_pdf('/path/to/mydata/pdf'))
        self.assertFalse(is_pdf('pdf'))

    def test_is_valid(self):
        self.assertTrue(is_valid('/path/to/mydata/file.jpg'))
        self.assertTrue(is_valid('/path/to/mydata/file.Jpg'))
        self.assertTrue(is_valid('/path/to/mydata/file.jPeg'))
        self.assertTrue(is_valid('/path/to/mydata/file.PNG'))
        self.assertTrue(is_valid('/path/to/mydata/file.pdf'))
        self.assertTrue(is_valid('/path/to/mydata/file.docx.jpg'))
        self.assertFalse(is_valid('/path/to/mydata/file.jpg.docx'))
        self.assertFalse(is_valid('/path/to/mydata/file.gif'))
        self.assertFalse(is_valid('jpg'))
