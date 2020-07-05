"""
Test case for step enum.
"""

import os
import unittest
from billys.train import Step


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class StepTest(unittest.TestCase):
    def test_step_range(self):
        """
        Steps are totally ordered.
        """
        self.assertTrue(Step.INIT < Step.DEWARP)
        self.assertTrue(Step.DEWARP < Step.CONTRAST)
        self.assertTrue(Step.CONTRAST < Step.ORC)
        self.assertTrue(Step.ORC < Step.FEAT_PREPROC)
        self.assertTrue(Step.FEAT_PREPROC < Step.TRAIN_CLASSIFIER)

    def test_to_step(self):
        """
        Integer can be converted to steps, and the conversion is right.
        """
        self.assertEqual(Step(0), Step.INIT)
        self.assertEqual(Step(1), Step.DEWARP)
        self.assertEqual(Step(2), Step.CONTRAST)
        self.assertEqual(Step(3), Step.ORC)
        self.assertEqual(Step(4), Step.FEAT_PREPROC)
        self.assertEqual(Step(5), Step.TRAIN_CLASSIFIER)

    def test_to_int(self):
        """
        Steps can be converted to int, and the conversion is right.
        """
        self.assertEqual(int(Step.INIT), 0)
        self.assertEqual(int(Step.DEWARP), 1)
        self.assertEqual(int(Step.CONTRAST), 2)
        self.assertEqual(int(Step.ORC), 3)
        self.assertEqual(int(Step.FEAT_PREPROC), 4)
        self.assertEqual(int(Step.TRAIN_CLASSIFIER), 5)
