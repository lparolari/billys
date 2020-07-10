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


def get_default_steps() -> List[str]:
    """
    Returns
    -------
    steps
        The list of the default step to perform for a full pipeline.
    """
    return [
        'fetch-billys',
        'init-dataframe',
        'print',
        'dewarp',
        'rotation',
        'brightness',
        'contrast',
        'ocr',
        'show-boxed-text',
        'extract-text',
        'preprocess-text',
        'save-dump',
        # TODO: complete pipeline
    ]


def make_config(custom={}):
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

        Notation: For simplicity dict keys will be flattened in docs.

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
        The given configuration dict merged with defaults.
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
        }
    }

    # Merge given config with defaults.
    return deepmerge.always_merger.merge(default, custom)


def make_steps(step_list: List[str] = get_default_steps(), config=make_config()):
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
        # 'fetch-checkpoint': lambda *_: fetch(**config.get('fetch-checkpoint')),
        'fetch-dump': lambda *_: revert(**config.get('fetch-dump')),
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
        'train-classifier': train_classifier
    }

    to_do_steps = []

    for step in (step_list):
        func = available_steps.get(step)
        if func is not None:
            to_do_steps.append(tuple((step, func)))
        else:
            logging.warning(f'Unrecognized step {step}, skipping.')

    return to_do_steps


def get_available_steps(config=make_config()) -> List[str]:
    """
    Returns
    -------
    steps
        The list of all available steps in pipeline.
    """
    return list(map(lambda x: x[0], make_steps(config=config)))


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
