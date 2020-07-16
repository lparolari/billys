"""
Module implementing the pipeline.

Plese use only **kebab-cased** idenfiers for pipeline steps
(https://it.wikipedia.org/wiki/Kebab_case). Other types of case could
compromise the saving and loading functions for checkpoints.
"""

import logging
import os
from typing import List, Optional

import cv2
import deepmerge
import pandas as pd
import piexif
import sklearn
from PIL import Image, ImageEnhance, ImageOps

import billys.constant as billys_const
from billys.steps import (brightness, build_dataframe, classify, contrast,
                          convert_to_images, dewarp, dump, extract_text,
                          fetch_dataset, fetch_filenames, log, ocr,
                          preprocess_text, revert, rotation, show_boxed_text,
                          skip, train_classifier)
from billys.util import (get_data_home, get_elapsed_time, get_target_names,
                         identity, now)

# TODO: add step management in pipelines.
# TODO: add config management in pipelines.


def train_naive_bayes(
    dataset_name,
    data_home=None,
    force_good=True,
    homography_model_path=os.path.join(
        os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
    steps=['convert_to_images', 'dewarp', 'rotation', 'brightness', 'contrast',
           'ocr', 'show_boxed_text', 'extract_text', 'preprocess_text']):
    """
    Perform the training of a Naive Bayes classifier from dataset
    and return the trained classifier.
    """

    billys_const.USE_DATASET_STRUCTURE = True

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    datasets = fetch_dataset(name=dataset_name, data_home=data_home)
    target_names = get_target_names(datasets)

    df = build_dataframe(
        datasets, input_type='datasets', force_good=force_good)

    # Performing pipeline
    if 'convert_to_images' in steps:
        df = convert_to_images(df)
    if 'dewarp' in steps:
        df = dewarp(df, homography_model_path=homography_model_path)
    if 'rotation' in steps:
        df = rotation(df)
    if 'brightness' in steps:
        df = brightness(df)
    if 'contrast' in steps:
        df = contrast(df)
    if 'ocr' in steps:
        df = ocr(df)
    if 'show_boxed_text' in steps:
        df = show_boxed_text(df)
    if 'extract_text' in steps:
        df = extract_text(df)
    if 'preprocess_text' in steps:
        df = preprocess_text(df)

    # TODO: remove this in future...
    dump(df, name='preprocessed_dataset.pkl')

    clf = train_classifier(df)

    return clf


def classify_from_filenames(
    filenames,
    classifier,
    target_names,
    data_home=None,
    force_good=True,
    homography_model_path=os.path.join(
        os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
    steps=['convert_to_images', 'dewarp', 'rotation', 'brightness', 'contrast',
           'ocr', 'show_boxed_text', 'extract_text', 'preprocess_text']):
    """
    Perform data classification with classifier `classifier` from
    given filenames, without verifying targets. No metrics shown.
    """

    billys_const.USE_DATASET_STRUCTURE = False

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    filenames = fetch_filenames(filenames, data_home=data_home)

    df = build_dataframe(
        filenames, input_type='filenames', force_good=force_good)

    # Performing pipeline
    if 'convert_to_images' in steps:
        df = convert_to_images(df)
    if 'dewarp' in steps:
        df = dewarp(df, homography_model_path=homography_model_path)
    if 'rotation' in steps:
        df = rotation(df)
    if 'brightness' in steps:
        df = brightness(df)
    if 'contrast' in steps:
        df = contrast(df)
    if 'ocr' in steps:
        df = ocr(df)
    if 'show_boxed_text' in steps:
        df = show_boxed_text(df)
    if 'extract_text' in steps:
        df = extract_text(df)
    if 'preprocess_text' in steps:
        df = preprocess_text(df)

    # Performing prediction
    predicted = classify(df, classifier)

    logging.info(f'Target names: {target_names}')

    # Showing perdicted
    for i in range(len(filenames)):
        filename = filenames[i]
        target = predicted[i]
        target_name = target_names[target]

        print(f'{filename} => {target_name}')

    return predicted


def classify_from_dataset(
    dataset_name,
    classifier,
    target_names,
    data_home=None,
    force_good=True,
    homography_model_path=os.path.join(
        os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
    steps=['convert_to_images', 'dewarp', 'rotation', 'brightness', 'contrast',
           'ocr', 'show_boxed_text', 'extract_text', 'preprocess_text']):
    """
    Perform data classification with classifier `classifier` from given
    dataset. Perform also test phase and show metrics.
    """
    billys_const.USE_DATASET_STRUCTURE = True

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    datasets = fetch_dataset(dataset_name, data_home=data_home)

    df = build_dataframe(
        datasets, input_type='datasets', force_good=force_good)

    # Performing pipeline
    if 'convert_to_images' in steps:
        df = convert_to_images(df)
    if 'dewarp' in steps:
        df = dewarp(df, homography_model_path=homography_model_path)
    if 'rotation' in steps:
        df = rotation(df)
    if 'brightness' in steps:
        df = brightness(df)
    if 'contrast' in steps:
        df = contrast(df)
    if 'ocr' in steps:
        df = ocr(df)
    if 'show_boxed_text' in steps:
        df = show_boxed_text(df)
    if 'extract_text' in steps:
        df = extract_text(df)
    if 'preprocess_text' in steps:
        df = preprocess_text(df)

    # Performing prediction
    predicted = classify(df, classifier)

    logging.info(f'Target names: {target_names}')

    filenames = df['filename'].tolist()

    # Print predicted values.
    for i in range(len(filenames)):
        filename = filenames[i]  # df.loc[i]['filename']  # filenames[i]

        target = predicted[i]
        target_name = target_names[target]

        print(f'{filename} => {target_name}')

    y_true = df['target'].tolist()
    y_pred = predicted
    labels = sorted(df['target'].unique().tolist())

    # Show metrics.
    print(sklearn.metrics.classification_report(
        y_true,
        y_pred,
        target_names=[target_names[i] for i in labels],
        labels=labels,
    ))

    return predicted


def classify_from_dump(
    dump_name,
    classifier,
    target_names,
    data_home=None,
    force_good=True,
    homography_model_path=os.path.join(
        os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
    steps=[]  # steps are empty because we assume by default that the dump has been already processed
):
    """
    Perform data classification with classifier `classifier` from given
    dataset dump. Perform also test phase and show metrics.
    """
    billys_const.USE_DATASET_STRUCTURE = True

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    df = revert(name=dump_name)

    # Performing pipeline
    if 'convert_to_images' in steps:
        df = convert_to_images(df)
    if 'dewarp' in steps:
        df = dewarp(df, homography_model_path=homography_model_path)
    if 'rotation' in steps:
        df = rotation(df)
    if 'brightness' in steps:
        df = brightness(df)
    if 'contrast' in steps:
        df = contrast(df)
    if 'ocr' in steps:
        df = ocr(df)
    if 'show_boxed_text' in steps:
        df = show_boxed_text(df)
    if 'extract_text' in steps:
        df = extract_text(df)
    if 'preprocess_text' in steps:
        df = preprocess_text(df)

    # Performing prediction
    predicted = classify(df, classifier)

    logging.info(f'Target names: {target_names}')

    filenames = df['filename'].tolist()

    # Print predicted values.
    for i in range(len(filenames)):
        filename = filenames[i]  # df.loc[i]['filename']  # filenames[i]

        target = predicted[i]
        target_name = target_names[target]

        print(f'{filename} => {target_name}')

    y_true = df['target'].tolist()
    y_pred = predicted
    labels = sorted(df['target'].unique().tolist())

    # Show metrics.
    print(sklearn.metrics.classification_report(
        y_true,
        y_pred,
        target_names=[target_names[i] for i in labels],
        labels=labels,
    ))

    return predicted


def get_classifier(use_deterministic: bool = False, classifier_dump_name: str = None, data_home: str = None):
    if not use_deterministic:
        return revert(name=classifier_dump_name, data_home=data_home)
    else:
        class DeterministicClassifier:
            def predict(self, documents: List[str]) -> List[int]:
                from billys.text.classification import classify_bow

                return classify_bow(documents=documents, unknown_cat_id=5, max_word_diff=1)

        return DeterministicClassifier()
