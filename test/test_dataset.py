"""
Test case for dataset module.
"""

import os
import unittest
from pickle import load

from billys.dataset import save, revert

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class DatasetTest(unittest.TestCase):

    def test_save(self):
        expected = dict(a=1, b=[1, 2, 3], c='foo', d=dict(e='1', f=2))
        name = save(step=0, obj=expected, data_home='/tmp')
        actual = load(open(os.path.join('/tmp', name), 'rb'))

        self.assertEqual(expected, actual)

    def test_revert(self):
        save(step=0, obj=dict(foo=1), data_home='/tmp')
        save(step=1, obj="foo", data_home='/tmp')
        save(step=0, obj=2, data_home='/tmp')
        save(step=0, obj=dict(foo="bar"), data_home='/tmp')  # expected
        save(step=2, obj=["bar", "baz"], data_home='/tmp')

        actual = revert(0, data_home='/tmp')

        self.assertEqual(dict(foo="bar"), actual)
