"""
Test case for cli.
"""

import os
import unittest

from billys.pipeline import PresetConfig, get_config, get_steps, make_steps, pipeline
from billys.util import get_data_home

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.zero = ('zero', lambda *_: 0)
        self.inc = ('inc', lambda x: x + 1)

    def test_preset_config(self):
        self.assertEqual(
            PresetConfig.PRECOMPILED_STEPS['preprocess_train_dataset'],
            PresetConfig('preprocess_train_dataset').get_steps())
        self.assertEqual(
            PresetConfig.PRECOMPILED_CONFIG['preprocess_train_dataset'],
            PresetConfig('preprocess_train_dataset').get_config())
        self.assertEqual(
            {'fetch-train-test-dump': {'train': {'name': 'train_df.pkl'}, 'test': {'name': 'test_df.pkl'}},
             'save-dump': {'name': f'my_train_df.pkl'}},
            PresetConfig('do_train').get_config({'save-dump': {'name': f'my_train_df.pkl'}}))
        self.assertRaises(
            ValueError,
            lambda: PresetConfig('myconfig'))

    def test_get_steps(self):
        self.assertEqual([], get_steps(steps=[]))
        self.assertEqual(['fetch-billys', 'init-dataframe-from-dataset'],
                         get_steps(steps=['fetch-billys', 'init-dataframe-from-dataset']))
        self.assertEqual(
            PresetConfig('preprocess_train_dataset').get_steps(), get_steps())

    def test_get_config(self):
        self.assertEqual({
            'fetch-billys': {},
            'fetch-dump': {},
            'save-dump': {'name': 'train_df.pkl'},
            'fetch-data-and-classifier': {},
            'init-dataframe-from-dataset': {},
            'init-dataframe-from-filenames': {},
            'dewarp': {'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')},
            'fetch-train-test-dump': {'train': {}, 'test': {}}
        }, get_config())

        self.assertEqual({
            'fetch-billys': {
                'data_home': '/path/to/datahome',
                'name': 'my-dataset',
                'subset': 'test',
            },
            'fetch-dump': {
                'name': 'my-dump.pkl',
            },
            'save-dump': {},
            'init-dataframe-from-dataset': {},
            'init-dataframe-from-filenames': {},
            'fetch-data-and-classifier': {},
            'dewarp': {
                'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')
            },
            'fetch-train-test-dump': {
                'train': {},
                'test': {}
            }
        }, get_config(custom={
            'fetch-billys': {
                'data_home': get_data_home('/path/to/datahome'),
                'name': 'my-dataset',
                'subset': 'test',
            },
            'fetch-dump': {
                'name': 'my-dump.pkl',
            },
            'init-dataframe-from-dataset': {},
            'dewarp': {
                'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')
            }
        }))

    def test_make_steps(self):
        self.assertEqual(
            PresetConfig('preprocess_train_dataset').get_steps(),
            list(map(lambda x: x[0], make_steps())))

        self.assertEqual(
            [],
            list(map(lambda x: x[0], make_steps(step_list=[]))))

        self.assertEqual(
            ['print', 'dewarp', 'ocr'],
            list(map(lambda x: x[0], make_steps(step_list=['print', 'dewarp', 'ocr']))))

        self.assertEqual(
            ['print', 'dewarp', 'ocr'],
            list(map(lambda x: x[0], make_steps(step_list=['print', 'dewarp', 'ocr', 'ocr_test']))))

    def test_pipeline(self):
        self.assertEqual(None, pipeline(steps=[]))
        self.assertEqual(
            3,
            pipeline(steps=[self.zero, self.inc, self.inc, self.inc]))
