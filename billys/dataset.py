"""
Manage dataset interaction.
"""

import os
from datetime import datetime

from pickle import dump
from sklearn.datasets import load_files

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

    cache = dict(train=load_files(train_path, **params),
                 test=load_files(test_path, **params))

    if subset in ('train', 'test'):
        return cache[subset]
    elif subset == 'all':
        return cache
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)


def save(step: int, obj, data_home=None):
    """
    Save the given object as checkpoint. 

    Parameters
    ----------
    step: int, required
        The checkpoint step.

    data_home : optional, default: None
        A path for checkpoints object. 
        If None, `DEFAULT_PATH/checkpoints` will be used.
    """
    now = datetime.now()
    now_str = now.strftime('%Y%m%d-%H%M%S-%f')

    if data_home is None:
        path = os.path.join(get_data_home(), 'checkpoints')
    else:
        path = data_home
    name = f'{now_str}_step-{step}.pkl'

    dump(obj, open(f'{os.path.join(path, name)}', 'wb'))

    return name


def load(step: int, data_home=None):
    """
    Load the dataset checkpoint for the step `step`.

    Parameters
    ----------
    step: int, required
        Load the checkpoint for the given step.

    data_home : optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all billys data is stored in default subfolder.
    """
    raise NotImplementedError
