"""
Module implementing the pipeline.

Plese use only **kebab-cased** idenfiers for pipeline steps
(https://it.wikipedia.org/wiki/Kebab_case). Other types of case could
compromise the saving and loading functions for checkpoints.
"""

import logging
import os
from typing import Optional, List

import cv2
import pandas as pd
import piexif
from PIL import Image, ImageEnhance, ImageOps
import deepmerge

from billys.steps import dump, revert, show, skip
from billys.steps import build, fetch
from billys.steps import brightness, contrast, dewarp, rotation
from billys.steps import ocr, show_boxed_text
from billys.steps import extract_text, preprocess_text
from billys.steps import train_classifier
from billys.util import get_elapsed_time, now, get_data_home, identity


def merge_config(c1, c2):
    """
    Merge configs `c1` and `c2` with :func:`deepmerge.always_merger.merge`
    and return the merged configuration.

    Returns
    -------
    config
        Merged configuration
    """
    return deepmerge.always_merger.merge(c1, c2)


class PresetConfig:
    """
    Class for preset steps and config.
    """

    AVAILABLE_STAGES = [
        'preprocess_train_dataset',
        'preprocess_test_dataset',
        'do_train',
    ]

    PRECOMPILED_STEPS = {
        'preprocess_train_dataset': [
            'fetch-billys',
            'init-dataframe',
            'dewarp',
            'rotation',
            'brightness',
            'contrast',
            'ocr',
            'show-boxed-text',
            'extract-text',
            'preprocess-text',
            'save-dump',
        ],
        'preprocess_test_dataset': [
            'fetch-billys',
            'init-dataframe',
            'dewarp',
            'rotation',
            'brightness',
            'contrast',
            'ocr',
            'show-boxed-text',
            'extract-text',
            'preprocess-text',
            'save-dump',
        ],
        'do_train': ['fetch-train-test-dump', 'train-classifier', 'save-dump']
    }

    PRECOMPILED_CONFIG = {
        'preprocess_train_dataset':  {
            'save-dump': {
                'name': f'train_df.pkl'
            }
        },
        'preprocess_test_dataset': {
            'save-dump': {
                'name': f'test_df.pkl'
            }
        },
        'do_train': {
            'fetch-train-test-dump': {
                'train': {
                    'name': 'train_df.pkl'
                },
                'test': {
                    'name': 'test_df.pkl'
                }
            },
            'save-dump': {
                'name': 'trained_classifier.pkl'
            }
        },
    }

    def __init__(self, stage: str):
        if stage not in self.AVAILABLE_STAGES:
            raise ValueError(
                f'The stage {stage} is not valid, it should be one of {self.AVAILABLE_STAGES}')
        self.stage = stage

    def get_steps(self):
        return self.PRECOMPILED_STEPS[self.stage]

    def get_config(self, custom={}):
        return merge_config(self.PRECOMPILED_CONFIG[self.stage], custom)


def get_steps(steps=(PresetConfig(stage='preprocess_train_dataset').get_steps())):
    """
    Get pipeline steps.

    Parameters
    ----------
    steps
        A list of steps. By default steps for 'preprocess_train_dataset' build with 
        `PresetConfig` are returned.

    Returns
    -------
    steps
        Given steps or a default preset.
    """
    # TODO: check steps validity
    return steps


def get_config(custom=(PresetConfig(stage='preprocess_train_dataset').get_config())):
    """
    Get the configuration for pipeline steps.

    Parameters
    ----------
    config
        A dict, with the following structure
        ```
        {
            'step-1': {
                'param-1-1': 'value-1-1',
                'param-1-2': 'value-1-2',
                ...
            }
            'step-2': {
                'param-2-1': 'value-2-1',
                'param-2-2': 'value-2-2',
                ...
            }
            ...
        }
        ```

        Notation: For simplicity dict keys will be flattened.

        Available configurations:

        fetch-billys: dict, default={}
            Configuration for fetch-billys step.
            Available parameters are
             * data_home:                               str, optional
             * name:                                    str, optional
             * subset:                                  str, optional

        fetch-dump: dict, default={}
            Configuration for fetch-dump step.
            Available parameters
            * data_home:                                str, optional
            * name:                                     str, required

        save-dump: dict, default={}
            Configuration for fetch-dump step.
            Available parameters
            * data_home:                                str, optional
            * name:                                     str, required

        init-dataframe: dict, default={}
            Configuration for init-dataframe step.
            Available parameters
            * force_good:                              bool, optional
            * subset:                                   str, optional

        dewarp: dict, default={}
            Configuration for dewarp step.
            Available parameters
            * homography_model_path:                    str, optional

        fetch-train-test-dump: dict, default={}
            Configuration for train and test dump to load.
            Available parameters
            * train.name                                str, required
            * test.name                                 str, required

        Please for further details refer to steps functions documentation.

        Example:
        ```
        {
            'fetch-billys': {
                'data_home': get_data_home('/path/to/datahome/foo'),
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
        ```

        Note that you can overwrite only required keys and steps.
        If keys or steps are omitted the will assume default values

    Returns
    -------
    config
        The given configuration dict merged with defaults or a default preset.
        Given configurations overwrite defaults.
    """

    # Default configs
    default = {
        'fetch-billys': {},
        'fetch-dump': {},
        'save-dump': {},
        'init-dataframe': {},
        'dewarp': {
            'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')
        },
        'fetch-train-test-dump': {
            'train': {},
            'test': {}
        }
    }

    # Merge given config with defaults.
    return deepmerge.always_merger.merge(default, custom)


def make_steps(step_list: List[str] = get_steps(), config=get_config()):
    """
    Build a list of pairs where the first component is the step name, while the
    second component is the function to run for that step.

    Returns
    -------
    to_do_steps
        A list of step.
    """

    logging.debug(f'Building steps {step_list} with config {config}')

    available_steps = {
        'fetch-billys': lambda *_: fetch(**config.get('fetch-billys')),
        'fetch-dump': lambda *_: revert(**config.get('fetch-dump')),
        'fetch-train-test-dump': lambda *_: tuple([revert(**config.get('fetch-train-test-dump').get('train')), revert(**config.get('fetch-train-test-dump').get('test'))]),
        'save-dump': lambda *x: dump(*x, **config.get('save-dump')),
        'init-dataframe': lambda *x: build(*x, **config.get('init-dataframe')),
        'print': lambda *x: show(*x),
        'dewarp': lambda *x: dewarp(*x, **config.get('dewarp')),
        'rotation': rotation,
        'brightness': brightness,
        'contrast': contrast,
        'ocr': ocr,
        'show-boxed-text': show_boxed_text,
        'extract-text': extract_text,
        'preprocess-text': preprocess_text,
        'train-classifier': train_classifier,
    }

    to_do_steps = []

    for step in (step_list):
        func = available_steps.get(step)
        if func is not None:
            to_do_steps.append(tuple((step, func)))
        else:
            logging.warning(f'Unrecognized step {step}, skipping.')

    return to_do_steps


def pipeline(steps):
    """
    Run the training pipeline.

    Parameters
    ----------
    steps
        List of steps, i.e., pairs (name, func).
    """

    out = None
    i = 0

    start_time = now()

    for item in steps:
        step, func = item
        logging.info(f'Performing step {i}: {step} ... ')

        prev_out = out
        out = func(prev_out)

        i += 1

    end_time = now()
    elapsed = get_elapsed_time(start_time, end_time)

    logging.info(f'Pipeline completed in {elapsed} seconds.')

    return out
