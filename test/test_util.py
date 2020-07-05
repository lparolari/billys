"""
Test case for util module.
"""

import os
import unittest
from billys.util import get_data_home, BILLYS_WORKSPACE_NAME


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class UtilTest(unittest.TestCase):
    def test_get_data_home_none(self):
        self.assertEqual(get_data_home(), os.path.join(
            os.environ['HOME'], BILLYS_WORKSPACE_NAME))

    def test_get_data_home_given(self):
        self.assertEqual(get_data_home('/path/to/billys/home'),
                         '/path/to/billys/home')
