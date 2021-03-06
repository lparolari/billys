"""
Test case for pipeline module.
"""

import os
import unittest

import pandas as pd

from billys.steps import build_dataframe_from_dataset, build_dataframe_from_filenames, fetch_dataset, fetch_filenames, revert
from billys.steps import show, skip

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class StepsTest(unittest.TestCase):

    def setUp(self):
        # mock dataframe
        d = {'a': ['a1', 'a2'], 'b': ['b1', 'b2'], 'c': ['c1', 'c2']}
        df = pd.DataFrame(data=d)

        self.df = df

        # mock dataset
        self.filenames = ['/tmp/billys/train/cat1/1.png',
                          '/tmp/billys/test/cat1/1.png']
        for filename in self.filenames:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'w+') as f:
                f.write('example')
                f.flush()

    def test_show(self):
        self.assertTrue(show(self.df).equals(self.df))

    def test_skip(self):
        self.assertTrue(skip(self.df).equals(self.df))

    def test_fetch_dataset(self):
        dataset = fetch_dataset('/tmp')
        self.assertEqual(dataset.filenames, ['/tmp/billys/train/cat1/1.png'])

    def test_build_dataframe_from_dataset(self):
        dataset = fetch_dataset('/tmp')
        df = build_dataframe_from_dataset(dataset, True)
        df.equals(pd.DataFrame(data={
                  'filename': ['/tmp/billys/train/cat1/1.png'],
                  'target': [0],
                  'data': ['example'],
                  'is_grayscale': [False],
                  'smart_doc': [False],
                  'good': [True],
                  'is_pdf': [False]}))

        # TODO: dewarp
        # TODO: contrast
        # TODO: ocr
        # TODO: feat extr.
        # TODO: classification
