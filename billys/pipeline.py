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


def pipeline(data_home: str = os.path.join(os.getcwd(), 'dataset'),
             homography_model_path: str = os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
             force_good: bool = True):
    """
    Run the training pipeline.

    Parameters
    ----------
    TODO
    """

    # Plese use only **kebab-cased** idenfiers for pipeline steps
    # (https://it.wikipedia.org/wiki/Kebab_case)
    # Other types of case could compromise the saving and loading
    # functions for checkpoints.

    steps = [
        ('fetch-billys', lambda *_: fetch(data_home)),
        ('init-dataframe', lambda dataset: build(dataset, force_good)),
        ('print', show),
        ('dewarp', lambda df: dewarp(df, homography_model_path)),
        ('rotation', rotation),
        ('brightness', brightness),
        ('contrast', contrast),
        ('ocr', ocr),
        ('show-boxed-text', show_boxed_text),
        # ('dump-ocr', dump),
        # ('print', show),
        # ('feat-preproc', skip),
        # ('train-classifier', skip)
    ]

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
