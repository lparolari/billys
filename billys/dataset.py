"""
Manage dataset interaction.
"""

import logging
import os
import pathlib
from typing import List, Dict

import cv2
import pandas as pd
import sklearn
import sklearn.datasets

from billys.constant import BILLYS_SUPPORTED_IMAGES_FILE_LIST
from billys.util import ensure_dir, get_data_home, get_data_tmp, make_dataset_filename


def fetch_billys(data_home=None,
                 name='billys',
                 subset='train',
                 description=None,
                 categories=None,
                 load_content=False,  # Note: in scikit this is True by default!
                 shuffle=True,
                 encoding=None,
                 decode_error='strict',
                 random_state=0):
    """
    Load the billys dataset.

    Parameters
    ----------
    data_home : optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all billys data is stored in default subfolder.

    name: optiona, default: 'billys'
        The dataset name.

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    Returns
    -------
    dataset
        If `subset` is one of 'train', 'test' then the dataset in `scikit.utils.Bunch` is returned.
        Otherwise it is returned a dict with key train and test and values are the two subset
        with `scikit.utils.Bunch` format.
    """

    data_home = get_data_home(data_home=data_home)
    target_dir = os.path.join(data_home, name)

    train_path = os.path.join(target_dir, 'train')
    test_path = os.path.join(target_dir, 'test')

    params = dict(
        description=description,
        categories=categories,
        load_content=load_content,
        shuffle=shuffle,
        encoding=encoding,
        decode_error=decode_error,
        random_state=random_state
    )

    cache = dict(train=sklearn.datasets.load_files(train_path, **params),
                 test=sklearn.datasets.load_files(test_path, **params))

    if subset in ('train', 'test'):
        return cache[subset]
    elif subset == 'all':
        return cache
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)


def make_dataframe_from_datasets(datasets: Dict[str, sklearn.utils.Bunch],
                                 force_good: bool = False) -> pd.DataFrame:
    """
    Create a dataframe from the given dataset.

    Parameters
    ----------
    dataset: scikit.utils.Bunch, required
        See https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html

    force_good: bool, optional, default: False
        If True all dataset samples are marked as good, and they will skip some
        pipeline steps like dewarping an contrast augmentation.

    Returns
    -------
    df
        A new dataframe with the following columns
         * 'filename', the path to image or pdf file,
         * 'target', the encoded values for target names,
         * 'target_name', the target name,
         * 'is_grayscale', whether the image is in greyscale,
         * 'is_good', whether the image is good, i.e., it does not require dewarping,
         * 'is_pdf', whether the image file is in pdf format,
         * 'subset', the dataset subset, one in 'train', 'test'.
    """
    df = pd.DataFrame(columns=['filename', 'target', 'target_name', 'is_grayscale',
                               'is_good', 'is_pdf', 'subset', 'is_valid'])

    for subset, dataset in datasets.items():
        subset_df = pd.DataFrame(data={
            'filename': dataset.filenames,
            'target': dataset.target,
            'target_name': [dataset.target_names[target] for target in dataset.target],
            'is_grayscale': [False for _ in dataset.filenames],
            'is_good': [is_good(filename, force_good) for filename in dataset.filenames],
            'is_pdf': [is_pdf(filename) for filename in dataset.filenames],
            'is_valid': [is_valid(filename) for filename in dataset.filenames],
            'subset': [subset for _ in dataset.filenames]
        })
        df = df.append(subset_df, ignore_index=True)

    drop_invalid_values(df)
    drop_isvalid_column(df)

    return df


def make_dataframe_from_filenames(filenames: List[str], force_good: bool = False):
    df = pd.DataFrame(
        columns=['filename', 'is_grayscale', 'is_pdf', 'is_good'])

    df['filename'] = filenames
    df['is_grayscale'] = [is_grayscale(filename) for filename in filenames]
    df['is_pdf'] = [is_pdf(filename) for filename in filenames]
    df['is_good'] = [is_good(filename, force_good) for filename in filenames]

    # addiotional column for validating files and discard the not valid ones
    df['is_valid'] = [is_valid(filename) for filename in filenames]

    drop_invalid_values(df)
    drop_isvalid_column(df)

    return df


def is_good(filename: str, force: bool = False) -> bool:
    """
    Verify whether a file is good or not, i.e., it does not need
    any shape or illumination modification. By default a file is 
    good if it is a pdf.

    Parameters
    ----------
    filename
        The file name.
    force
        Wether to force the goodness of a file.

    Returns
    -------
    good
        True whether the file is a pdf or its goodnes is forced,
        False otherwise.
    """
    return filename.endswith('.pdf') or force


def is_pdf(filename: str) -> bool:
    """
    Verify whether a file is a pdf or not.

    Parameters
    ----------
    filename
        The file name.

    Returns
    -------
    is_pdf
        True whether the file is a pdf, False otherwise.
    """
    return filename.lower().endswith('.pdf')


def is_valid(filename: str) -> bool:
    """
    Verify whether a file is good or not for the pipeline.

    Parameters
    ----------
    filename
        The file name.

    Returns
    -------
    is_valid
        True whether the file is a good file for the pipeline, i.e.,
        it is a supported file, False otherwise.
    """
    splitted = os.path.splitext(filename)

    if len(splitted) < 2:
        return False

    ext = splitted[1]
    ext = ext.lower()  # normalize
    ext = ext[1:]      # remove first dot (e.g. .pdf -> pdf)

    return ext in BILLYS_SUPPORTED_IMAGES_FILE_LIST


def is_grayscale(filename: str) -> bool:
    """
    Verify whether a file is in grayscale or not.

    Parameters
    ----------
    filename
        The file name.

    Returns
    -------
    is_grayscale
        True whether the file is in grayscale, False otherwise.
    """
    # TODO: automatic grayscale detection
    return False


def drop_invalid_values(df):
    """
    Drop inplace all the the invalid rows.
    """
    indexes = df[df['is_valid'] == False].index
    df.drop(indexes, inplace=True)


def drop_isvalid_column(df):
    """
    Drop the `is_valid` column inplace.
    """
    df.drop('is_valid', axis=1, inplace=True)


def make_filename(row, step: str = None, ext: str = None):
    """
    Build an image filename based on its stage. Possible stages
    are
        'training', 'classification'

    If the stage is 'training' a filename that respects the dataset
    structure is built, while if it is 'classification' a temporary
    filename is build.

    Parameters
    ----------
    row
        The dataframe row. Required columns are
            'stage', 'filename',
            + 'target_name', 'subset' if 'stage' is `training`
    step
        The pipeline step.

    Returns
    -------
    filename
        Filename based on stage.
    """
    from billys.constant import USE_DATASET_STRUCTURE

    if USE_DATASET_STRUCTURE:
        filename = row['filename']
        # TODO: remove this shitty if
        if ext is not None:
            new_filename = f'{os.path.splitext(filename)[0]}.{ext}'
        else:
            new_filename = filename
        return make_dataset_filename(
            filename=new_filename,
            cat=row['target_name'],
            subset=row['subset'],
            step=step)
    else:
        from datetime import datetime
        now = datetime.now()
        now_str = now.strftime('%Y%m%d-%H%M%S-%f')
        base_filename = os.path.basename(row['filename'])
        # TODO: remove this shitty if
        if ext is not None:
            new_filename = f'{os.path.splitext(base_filename)[0]}.{ext}'
        else:
            new_filename = base_filename
        return os.path.join(get_data_tmp(), step, new_filename)
