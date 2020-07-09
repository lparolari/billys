"""
Test case for cli.
"""

import os
import unittest

from billys.pipeline import get_default_steps, get_available_steps, make_config, make_steps, pipeline
from billys.util import get_data_home

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class PipelineTest(unittest.TestCase):

    def test_make_config_default(self):
        expected = {
            'fetch-billys': {},
            'fetch-dump': {},
            'save-dump': {},
            'init-dataframe': {},
            'dewarp': {}
        }

        self.assertEqual(expected, make_config())

    def test_make_config_overwrite(self):
        overwrite = {
            'fetch-billys': {
                'data_home': get_data_home('/path/to/datahome'),
                'name': 'my-dataset',
                'subset': 'test',
            },
            'fetch-dump': {
                'name': 'my-dump.pkl',
            },
            'init-dataframe': {},
            'dewarp': {
                'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')
            }
        }
        expected = {
            'fetch-billys': {
                'data_home': get_data_home('/path/to/datahome'),
                'name': 'my-dataset',
                'subset': 'test',
            },
            'fetch-dump': {
                'name': 'my-dump.pkl',
            },
            'save-dump': {},
            'init-dataframe': {},
            'dewarp': {
                'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')
            }
        }

        self.assertEqual(expected, make_config(custom=overwrite))

    def test_make_steps_no_args(self):
        expected = get_default_steps()
        actual = list(map(lambda x: x[0], make_steps()))
        self.assertEqual(expected, actual)

    def test_make_steps_empty_list(self):
        expected = []
        actual = list(map(lambda x: x[0], make_steps(step_list=[])))
        self.assertEqual(expected, actual)

    def test_make_steps_include_available_steps(self):
        step_list = ['print', 'dewarp', 'ocr']
        expected = step_list
        actual = list(map(lambda x: x[0], make_steps(
            step_list=step_list, config=make_config())))
        self.assertEqual(expected, actual)

    def test_make_steps_exclude_undefined_steps(self):
        step_list = ['print', 'dewarp', 'ocr', 'ocr_test']
        expected = ['print', 'dewarp', 'ocr']
        actual = list(map(lambda x: x[0], make_steps(
            step_list=step_list, config=make_config())))
        self.assertEqual(expected, actual)

    def test_default_in_available_steps(self):
        self.assertTrue(set(get_default_steps()).issubset(
            set(get_available_steps())))

    def test_pipeline_steps_empty_list(self):
        self.assertEqual(None, pipeline([]))

    def test_pipeline_data_propagation(self):
        zero = ('zero', lambda *_: 0)
        inc = ('inc', lambda x: x + 1)
        self.assertEqual(3, pipeline(steps=[zero, inc, inc, inc]))
