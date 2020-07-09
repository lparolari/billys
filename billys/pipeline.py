"""
Module implementing the pipeline.

Plese use only **kebab-cased** idenfiers for pipeline steps
(https://it.wikipedia.org/wiki/Kebab_case). Other types of case could
compromise the saving and loading functions for checkpoints.
"""

import logging
import os
from typing import List

import cv2
import pandas as pd
import piexif
from PIL import Image, ImageEnhance, ImageOps

from billys.steps import dump, show, skip
from billys.steps import build, fetch, pickle
from billys.steps import brightness, contrast, dewarp, rotation
from billys.steps import ocr, show_boxed_text
from billys.steps import extract_text, preprocess_text
from billys.steps import train_classifier
from billys.util import get_elapsed_time, now, get_data_home


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
        'ocr',
        'show-boxed-text',
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
                'param-1.1': 'value-1.1',
                'param-1.2': 'value-1.2',
                ...
            }
            'step-2': {
                'param-2.1': 'value-2.1',
                'param-2.2': 'value-2.2',
                ...
            }
            ...
        }
        ```
        You can omit a configuration for a pipeline step if you
        don't use it. If you use a pipeline step without specifying
        a configuration it will receive the dafault configuration.

        The default configuration is described below
        ```
        TODO
        ```

    Returns
    -------
    config
        The given configuration dict merged with defaults.
        Given configurations overwrite defaults.
    """

    # Default configs
    default = {
        'fetch-billys': {
            'data_home': get_data_home(),
        },
        'fetch-checkpoint': {
            'data_home': get_data_home(),
        },
        'fetch-dump': {
            'data_home': get_data_home(),
            'name': 'preprocessed.pkl',
        },
        'init-dataframe': {
            'force_good': True,
            'subset': 'train',
        },
        'dewarp': {
            'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')
        }
    }

    # Merge given config with defaults.
    return {**default, **custom}


def make_steps(step_list=None, config=make_config()):
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
        'fetch-checkpoint': lambda *_: fetch(**config.get('fetch-checkpoint')),
        'fetch-dump': lambda *_: pickle(**config.get('fetch-dump')),
        'save-dump': lambda *x: dump(*x),
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

    for step in (step_list or available_steps):
        to_do_steps.append(tuple((step, available_steps[step])))

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
