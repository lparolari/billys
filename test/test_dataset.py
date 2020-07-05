"""
Test case for dataset module.
"""

import os
import unittest
from pickle import load

from billys.dataset import save

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class DatasetTest(unittest.TestCase):

    def test_save(self):
        expected = dict(a=1, b=[1, 2, 3], c='foo', d=dict(e='1', f=2))
        name = save(step=0, obj=expected, data_home='/tmp')
        actual = load(open(os.path.join('/tmp', name), 'rb'))

        self.assertEqual(expected, actual)
