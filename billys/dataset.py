"""
Manage dataset interaction.
"""

import os
import pathlib
import logging

import cv2
import pandas as pd
import sklearn
import sklearn.datasets
from pdf2image import convert_from_path

from billys.util import get_data_home, get_data_tmp
from billys.constant import BILLYS_SUPPORTED_IMAGES_FILES


def fetch_billys(data_home=None,
                 subset='train',
                 description=None,
                 categories=None,
                 load_content=True,
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
    target_dir = os.path.join(data_home, 'billys')

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


def make_dataframe(dataset: sklearn.utils.Bunch, force_good=False):
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
         * 'grayscale', whether the image is in greyscale,
         * 'is_good', whether the image is good, i.e., it does not require dewarping,
         * 'is_pdf', whether the image file is in pdf format,
         * 'is_valid', whether the file can be used in pipeline or not.
    """
    df = pd.DataFrame(columns=['filename', 'target', 'grayscale', 'good', 'is_pdf', 'is_valid'])

    df['filename'] = dataset.filenames
    df['target'] = dataset.target
    df['grayscale'] = False
    df['is_good'] = [is_good(filename, force_good) for filename in dataset.filenames]
    df['is_pdf'] = [is_pdf(filename) for filename in dataset.filenames]
    df['is_valid'] = [is_valid(filename) for filename in dataset.filenames]

    return df


def read_file(df_row):
    """
    Read the file data if it is supported and return it. If the file
    is not supported, we ignore it.

    Parameters
    ----------
    df_row: 
        A dataframe row containing the image metadata.
        Required columns are
            'filename', 'is_valid', 'is_pdf'

    Returns
    -------
    imdata
        The image data encoded with cv2 format, i.e., a list with shape
        [w, h, number of channels].
    """
    
    filename = df_row['filename']
    is_valid = df_row['is_valid']
    is_pdf = df_row['is_pdf']

    if not is_valid:
        logging.warning(f'The file {filename} is not supported. Skipping.')
    else:
        if is_pdf

            # Convert pdf to image and the store the image data.
            pages = convert_from_path(filename)
            tmp_filename = os.path.join(get_data_tmp(), 'pdf.jpg')
            os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
            for page in pages:
                page.save(tmp_filename, 'JPEG')
                # As specification, we need only the first page.
                break
            return cv2.imread(tmp_filename)

        else:
            # Directly load the image data. We do not check the 
            # image format because we assume that it is valid.
            return cv2.imread(filename)


def is_good(filename: str, force: bool) -> bool:
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
    return filename.endswith('.pdf')


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
    return ext in BILLYS_SUPPORTED_IMAGES_FILE_LIST
