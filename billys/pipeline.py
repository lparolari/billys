"""
Module implementing the pipeline.

Plese use only **kebab-cased** idenfiers for pipeline steps
(https://it.wikipedia.org/wiki/Kebab_case). Other types of case could
compromise the saving and loading functions for checkpoints.
"""

import logging
import os

import cv2
import pandas as pd
import piexif
from PIL import Image, ImageEnhance, ImageOps

from billys.pipe.img_preproc import brightness, contrast, dewarp, rotation
from billys.pipe.init import build, fetch
from billys.pipe.ocr import ocr, show_boxed_text
from billys.pipe.shared import dump, show, skip
from billys.util import get_elapsed_time, now


"""Available pipeline stesp, to be combined and used with the :func:`pipeline`."""
PIPELINE_AVAILABLE_STEPS = {
    'fetch-billys': lambda *_: fetch(data_home=data_home),
    'fetch-checkpoint': lambda *_: fetch(data_home=data_home, name=dataset_name),
    'init-dataframe': lambda dataset: build(dataset, force_good=force_good),
    'print': show,
    'dewarp': lambda df: dewarp(df, homography_model_path=homography_model_path),
    'rotation': rotation,
    'brightness': brightness,
    'contrast': contrast,
    'ocr': ocr,
    'show-boxed-text': show_boxed_text,
}


def get_all_steps() -> List[str]:
    """
    Returns
    -------
    steps
        The list of all available steps in pipeline.
    """
    return PIPELINE_AVAILABLE_STEPS.keys()


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
        'fetch-checkpoint',
        'ocr',
        'show-boxed-text',
    ]


def get_default_config():
    """
    Returns
    -------
    config
        A dict with a default value for each config.
        Available configurations are
         * 'data_home'
         * 'dataset_name'
         * 'force_good'
         * 'homography_model_path'
    """
    return {
        'data_home': os.path.join(os.getcwd(), 'dataset'),
        'dataset_name': 'billys',
        'force_good': True,
        'homography_model_path': os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
    }


def make_steps(step_list, config=make_default_config()):
    """
    Build a list of pairs where the first component is the step name, while the
    second component is the function to run for that step.

    Returns
    -------
    to_do_steps
        A list of step.
    """
    data_home = config['data_home']
    dataset_name = config['dataset_name']
    force_good = config['force_good']
    homography_model_path = config['homography_model_path']

    to_do_steps = []

    for step in step_list:
        to_do_steps.append(tuple((step, PIPELINE_AVAILABLE_STEPS[step])))

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
