"""
Dataset initialization pipeline steps
"""
import typing
import os

import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageOps
import piexif

from billys.dataset import fetch_billys, make_dataframe
from billys.util import read_file, get_data_home


def fetch(data_home: typing.Optional[str] = None, name: str = 'billys', subset: str = 'train'):
    """
    Fetch the dataset from the path with logic in :func:`billys.util.get_data_home` and
    return it.

    Parameters
    ----------
    data_home: default: None
        The directory from which retrieve the dataset. See :func:`billys.util.get_data_home`.

    subset
        The subset of dataset to load. Can be one of 'train', 'test'.

    Returns
    -------
    dataset
        The dataset, see :func:`billys.dataset.fetch_billys`.
    """
    return fetch_billys(data_home=data_home, name=name, subset=subset)


def build(dataset, force_good: bool = False, subset: str = 'train') -> pd.DataFrame:
    """
    Initialize the dataframe from given dataset.

    Parameters
    ----------
    dataset: required
        The dataset loaded with :func:`billys.dataset.fetch_billys`.

    subset
        The subset of loaded dataset. Can be one of 'train', 'test'.

    force_good: default: False
        Force all the samples in the dataframe to be marked as good and skip
        some pipeline steps like dewarping and contrast aumentation.

    Returns
    -------
    df
        A new dataframe built with :func:`billys.dataset.make_dataframe`.
    """
    return make_dataframe(dataset=dataset, force_good=force_good, subset=subset)


def pickle(data_home: typing.Optional[str] = None, name: str = 'dataset.pkl'):
    return read_file(os.path.join(get_data_home(data_home=data_home), name), is_pkl=True)
