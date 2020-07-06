"""
Test case for dataset module.
"""

import os
import unittest
from pickle import load

from billys.checkpoint import Step, save, revert

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class DatasetTest(unittest.TestCase):

    def test_save(self):
        expected = dict(a=1, b=[1, 2, 3], c='foo', d=dict(e='1', f=2))
        name = save(step=Step.INIT, obj=expected, data_home='/tmp')
        actual = load(open(os.path.join('/tmp', name), 'rb'))

        self.assertEqual(expected, actual)

    def test_revert_step_greater_or_equal_one(self):
        save(step=Step.OCR, obj=dict(foo=1), data_home='/tmp')
        save(step=Step.DEWARP, obj="foo", data_home='/tmp')
        save(step=Step.OCR, obj=2, data_home='/tmp')
        save(step=Step.OCR, obj=dict(foo="bar"), data_home='/tmp')  # expected
        save(step=Step.CONTRAST, obj=["bar", "baz"], data_home='/tmp')

        actual = revert(Step(int(Step.OCR) + 1), data_home='/tmp')

        self.assertEqual(dict(foo="bar"), actual)

    def test_revert_step_less_or_equal_zero(self):
        self.assertRaises(AssertionError, lambda: revert(
            Step.INIT, data_home='/tmp'))
