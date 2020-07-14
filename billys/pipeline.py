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

from billys.steps import dump, revert, log, skip, convert_to_images
from billys.steps import build_dataframe, fetch_dataset, fetch_filenames
from billys.steps import brightness, contrast, dewarp, rotation
from billys.steps import ocr, show_boxed_text
from billys.steps import extract_text, preprocess_text
from billys.steps import train_classifier, classify
from billys.util import get_elapsed_time, now, get_data_home, identity, get_target_names
import billys.constant as billys_const

# TODO: add step management in pipelines.
# TODO: add config management in pipelines.


def train_pipeline(data_home=None,
                   dataset_name='billys',
                   force_good=True,
                   classifier_dump_name='trained_classifier.pkl',
                   targets_dump_name='target_names.pkl',
                   homography_model_path=os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')):

    billys_const.USE_DATASET_STRUCTURE = True

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    datasets = fetch_dataset(name=dataset_name, data_home=data_home)
    target_names = get_target_names(datasets)

    df = build_dataframe(
        datasets, input_type='datasets', force_good=force_good)

    df = convert_to_images(df)
    df = log(df)
    df = dewarp(df, homography_model_path=homography_model_path)
    df = rotation(df)
    df = brightness(df)
    df = contrast(df)
    df = ocr(df)
    df = extract_text(df)
    df = preprocess_text(df)

    clf = train_classifier(df)

    dump(target_names, name=targets_dump_name)
    dump(clf, name=classifier_dump_name, data_home=data_home)

    return clf


def classify_pipeline(data_home=None,
                      filenames='new_images',
                      force_good=True,
                      classifier_dump_name='trained_classifier.pkl',
                      targets_dump_name='target_names.pkl'):

    billys_const.USE_DATASET_STRUCTURE = False

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    clf = revert(name=classifier_dump_name, data_home=data_home)
    target_names = revert(name=targets_dump_name, data_home=data_home)

    filenames = fetch_filenames(filenames, data_home=data_home)

    df = build_dataframe(
        filenames, input_type='filenames', force_good=force_good)

    df = convert_to_images(df)
    # df = log(df)
    # df = dewarp(df, homography_model_path=homography_model_path)
    # df = rotation(df)
    # df = brightness(df)
    # df = contrast(df)
    df = ocr(df)
    df = extract_text(df)
    df = preprocess_text(df)

    predicted = classify(df, clf)

    return predicted
