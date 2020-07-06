"""
Manage dataset interaction.
"""

import os

import cv2
import pandas as pd
import sklearn
import sklearn.datasets

from billys.util import get_data_home


def fetch_billys(data_home=None,
                 subset='train',
                 description=None,
                 categories=None,
                 load_content=True,
                 shuffle=True,
                 encoding=None,
                 decode_error='strict',
                 random_state=0):
    """Load the billys dataset.

    Parameters
    ----------
    data_home : optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all billys data is stored in default subfolder.

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.
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


def make_dataframe(dataset: sklearn.utils.Bunch):
    """
    Create a dataframe from the given dataset.

    Parameters
    ----------
    dataset: scikit.utils.Bunch, required
        #sklearn.utils.Bunch
        See https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html
    """
    df = pd.DataFrame(columns=['filename', 'target',
                               'target_name', 'data', 'grayscale', 'smart_doc'])
    df['filename'] = dataset.filenames
    df['target'] = dataset.target
    df['target_name'] = dataset.target_names
    df['data'] = [cv2.imread(filename) for filename in dataset.filenames]
    df['grayscale'] = False
    df['smart_doc'] = False

    return df
