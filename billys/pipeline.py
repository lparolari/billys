import os

import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageOps
import piexif

from billys.pipe.shared import show, skip, dump 
from billys.pipe.init import fetch, build 
from billys.pipe.img_preproc import dewarp, rotation, brightness, contrast 
from billys.pipe.ocr import ocr, show_boxed_text


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

    for item in steps:
        step, func = item
        print(f'Performing step {i}: {step} ... ')

        prev_out = out
        out = func(prev_out)

        i += 1

    print('Pipeline completed.')

    return out
