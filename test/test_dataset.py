import logging
import os
import time
import unittest

import pandas as pd
from sklearn.utils import Bunch

from billys.dataset import fetch_billys, make_dataframe_from_dataset, make_dataframe_from_filenames, is_good, is_pdf, is_valid
from billys.util import sort

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class UtilTest(unittest.TestCase):

    def test_fetch_billys(self):
        # TODO
        pass

    def test_make_dataframe_from_dataset(self):
        dataset = Bunch(filenames=['/path/to/data/mydataset/train/cat1/1.jpg', '/path/to/data/mydataset/train/cat1/2.jpg',
                                   '/path/to/data/mydataset/train/cat2/1.jpg', '/path/to/data/mydataset/train/cat2/2.jpg'],
                        target=[0, 0, 1, 1],
                        target_names=['cat1', 'cat2'])
        df = make_dataframe_from_dataset(
            dataset=dataset, force_good=True, subset='train')

        # Checks

        self.assertEqual(
            sort(['stage', 'filename', 'target', 'target_name',
                  'is_grayscale', 'is_good', 'is_pdf', 'subset']),
            sort(df.columns.to_list()))                     # expected columns
        self.assertTrue((df['stage'] == 'training').all())  # expected stage
        self.assertTrue(len(df['filename'].tolist()) == 4)  # expected length
        self.assertTrue((df['subset'] == 'train').all())    # expected subset
        self.assertTrue((df['is_good'] == True).all()
                        )      # expected good (force)
        self.assertEqual(
            sort(['cat1', 'cat2']),
            sort(df['target_name'].unique().tolist()))      # expected target names

    def test_make_dataframe_from_filenames(self):
        filenames = ['/path/to/myimages/img1.png', '/img2.jpg',
                     '/path/to/otherfolder/mybill.pdf', 'path/to/invalid.docx']
        df = make_dataframe_from_filenames(
            filenames=filenames, force_good=True)

        self.assertEqual(
            sort(['stage', 'filename', 'is_grayscale', 'is_good', 'is_pdf']),
            sort(df.columns.to_list()))                           # expected columns
        self.assertTrue(len(df['filename'].tolist())
                        == 3)        # expected length
        # expected stage
        self.assertTrue((df['stage'] == 'classification').all())
        # expected good (force)
        self.assertTrue((df['is_good'] == True).all())
        self.assertEqual(
            df['is_pdf'].tolist(),
            [False, False, True])                                 # expected pdf
        self.assertTrue((df['is_grayscale'] == False).all()
                        )      # expected grayscale

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
