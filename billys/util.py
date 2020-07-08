import os
import os.path
import time
from decimal import ROUND_HALF_UP, Decimal

BILLYS_WORKSPACE_NAME = '.billys'


def get_data_home(data_home=None):
    """
    Returns
    ------- 
    path: str
        `data_home` if it is not None, otherwise a path to a directory into the 
        user HOME path.
    """

    if data_home is None:

        user_home = os.path.expanduser('~')
        if user_home is None:
            raise RuntimeError(
                'You should specify at least your home directory with HOME env variable.')

        return f'{os.path.join(user_home, BILLYS_WORKSPACE_NAME)}'

    return data_home


def get_data_tmp(data_tmp=None):
    """
    Returns
    ------- 
    path: str
        `data_tmp` if it is not None, otherwise a path to a directory tmp directory.
    """

    # To be system agnostic, for now, we create the tmp directory inside the data folder
    # obtained with `get_data_home` and appending the dir `.tmp`.

    if data_tmp is None:
        data_home = get_data_home(data_home=data_tmp)
        data_tmp = os.path.join(data_home, '.tmp')

    return data_tmp


def make_filename(filename, cat, subset, step, data_home=None):
    """
    Returns
    -------
    path
        A path to a file created with this parttern
            DATA_HOME/step/subset/cat/basename(filename)
    """
    name_ext = os.path.basename(filename)
    name_only = os.path.splitext(name_ext)[0]
    return os.path.join(get_data_home(data_home=data_home), step, subset, cat, f'{name_only}.jpg')


def ensure_dir(filename: str):
    """
    Ensures that the directory obtained with :func:`os.path.dirname`
    exists, and if not, create that directory.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def now():
    return time.time()


def get_elapsed_time(start_time, end_time):
    return Decimal(end_time - start_time).quantize(Decimal('.001'),
                                                   rounding=ROUND_HALF_UP)
