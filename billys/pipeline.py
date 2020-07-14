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

    from sklearn import metrics

    dump(target_names, name=targets_dump_name)
    dump(clf, name=classifier_dump_name, data_home=data_home)

    return clf


def classify_pipeline(data_home=None,
                      filenames='billys_warped',
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

    print(metrics.classification_report(
        df['target'].tolist(), predicted, target_names=target_names))
    metrics.confusion_matrix(df['target'], predicted)

    logging.info(f'Target names: {target_names}')

    for i in range(len(filenames)):
        filename = filenames[i]
        target_name = target_names[predicted[i]]
        print(f'{filename} => {target_name}')

    return predicted


def classify_2_pipeline(data_home=None,
                        dataset_name='billys_warped',
                        force_good=False,
                        classifier_dump_name='trained_classifier_DEFINITIVO.pkl',
                        targets_dump_name='target_names.pkl',
                        homography_model_path=os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5')):

    billys_const.USE_DATASET_STRUCTURE = True

    logging.debug(
        f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    clf = revert(name=classifier_dump_name, data_home=data_home)
    target_names = revert(name=targets_dump_name, data_home=data_home)

    # datasets = fetch_dataset(dataset_name, data_home=data_home)

    # df = build_dataframe(
    #     datasets, input_type='datasets', force_good=force_good)

    # df = convert_to_images(df)
    # # df = log(df)
    # df = dewarp(df, homography_model_path=homography_model_path)
    # df = rotation(df)
    # df = brightness(df)
    # df = contrast(df)
    # df = ocr(df)
    # df = extract_text(df)
    # df = preprocess_text(df)

    # dump(df, name='new_warped_bills_preprocessed.pkl')

    from sklearn import metrics

    df = revert('dataset_DEFINITIVO.pkl')

    df = extract_text(df)
    df = preprocess_text(df)

    predicted = classify(df, clf)

    logging.info(f'Target names: {target_names}')

    filenames = df['filename'].tolist()

    for i in range(len(df['filename'])):
        filename = filenames[i]  # df.loc[i]['filename']  # filenames[i]
        target_name = target_names[predicted[i]]
        print(f'{filename} => {target_name}')

    print(metrics.classification_report(
        df['target'].tolist(), predicted, target_names=target_names))
    # print(metrics.confusion_matrix(df['target'], predicted))

    return predicted


def train_deterministic_pipeline():
    """
    TODO
    """
    df = revert('dataset_DEFINITIVO.pkl')

    # df = show_boxed_text(df)
    df = extract_text(df)
    df = preprocess_text(df)

    from billys.steps import train_bow

    train_bow(df)


def classify_deterministic_pipeline(
    data_home=None,
    filenames='new_images',
    force_good=True,
    # classifier_dump_name='trained_classifier.pkl',
    # targets_dump_name='target_names.pkl'
):
    """
    TODO
    """
    # billys_const.USE_DATASET_STRUCTURE = False

    # logging.debug(
    #     f'USE_DATASET_STRUCTURE: {billys_const.USE_DATASET_STRUCTURE}')

    # clf = revert(name=classifier_dump_name, data_home=data_home)
    # target_names = revert(name=targets_dump_name, data_home=data_home)

    # filenames = fetch_filenames(filenames, data_home=data_home)

    # df = build_dataframe(
    #     filenames, input_type='filenames', force_good=force_good)

    # df = convert_to_images(df)
    # # df = log(df)
    # # df = dewarp(df, homography_model_path=homography_model_path)
    # df = rotation(df)
    # df = brightness(df, gain=1.0)
    # df = contrast(df, gain=1.0)
    # df = ocr(df)
    # df = show_boxed_text(df)
    # df = extract_text(df)
    # df = preprocess_text(df)

    df = revert('dataset_DEFINITIVO.pkl')

    df = extract_text(df)
    df = preprocess_text(df)

    for index, row in df.iterrows():
        if row['target'] == 0:

            print(row['ocr']['text'])
            print(row['text'])
            print()

    from billys.steps import classify_bow
    classify_bow(df)
