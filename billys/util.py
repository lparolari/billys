import cv2
import logging
import os
import os.path
import time
from decimal import ROUND_HALF_UP, Decimal
from pickle import load, dump
from typing import Any, Optional

from pdf2image import convert_from_path

BILLYS_WORKSPACE_NAME = '.billys'


def identity(x):
    return x


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


def get_filename(name, data_home=None):
    """
    Build a filename from its name and its path.
    You should provide file extension.

    Returns
    -------
    filename
        A file name with right path and name `name`.
    """
    return os.path.join(get_data_home(data_home), name)


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


def get_log_level(level: str) -> int:
    """
    Get logging level from string.
    """
    level = level.lower().strip()

    if level == 'critical':
        return logging.CRITICAL
    if level == 'fatal':
        return logging.FATAL
    if level == 'error':
        return logging.ERROR
    if level == 'warning' or level == 'warn':
        return logging.WARNING
    if level == 'info':
        return logging.INFO
    if level == 'debug':
        return logging.DEBUG

    return logging.NOTSET


def read_file(filename: str) -> Optional[str]:
    """
    Read a common file and return its content.

    Returns
    -------
    content
        The file content
    """
    try:
        with open(filename, 'r') as f:
            content = f.read
        return content
    except IOError as e:
        logging.error(f'Error reading file {filename}.')
        logging.error(e)


def save_file(content: str, filename: str) -> None:
    """
    Save a common file with content `content` to file `filename`.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
    except IOError as e:
        logging.error(f'Error writing file {filename}.')
        logging.error(e)


def read_dump(filename: str) -> Optional[Any]:
    """
    Read a dump file and return the python object.

    Returns
    -------
    object
        The dumped object
    """
    try:
        with open(filename, 'rb') as f:
            obj = load(f)
        return obj
    except IOError as e:
        logging.error(f'Error reading dump file {filename}.')
        logging.error(e)


def save_dump(obj: Any, filename: str) -> None:
    """
    Save a dump file with content `obj` to file `filename`.
    """
    try:
        with open(filename, 'wb') as f:
            dump(obj, f)
    except IOError as e:
        logging.error(f'Error saving dump file {filename}.')
        logging.error(e)


def read_image(filename: str, is_pdf: bool = False, engine: str = 'cv2'):
    """
    Read the file data if it is supported and return it. If the file
    is not supported, we ignore it.

    Parameters
    ----------
    filename
    is_pdf
    engine: cv2 or pil

    Returns
    -------
    imdata
        The image data encoded with cv2 format, i.e., a list with shape
        [w, h, number of channels].
    """
    if engine == 'cv2':
        if is_pdf:

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
    elif engine == 'pil':
        return Image.open(filename)
    else:
        loggin.warning(
            f'Supported engines are `cv2` or `pil`, you gived {engine}. Skipping.')
        return None


def save_image(filename, imdata, engine: str = 'cv2', dpi=None):
    """
    Save an image with content `imdata` to file `filename`.

    Parameters
    ----------
    filename
        Path to image file. If some directory containing the file
        is missing we create it.

    imdata
        The image content.

    engine
        The engine used to save the file.
        Can be one of `cv2` or `pil`. If other, do nothing.

    dpi
        A tuple specifying dpi. Valid only with `engine=pil`.
    """
    ensure_dir(filename)
    if engine == 'cv2':
        cv2.imwrite(filename, imdata)
    elif engine == 'pil':
        imdata.save(filename, 'jpeg', dpi=dpi)
    else:
        loggin.warning(
            f'Supported engines are `cv2` or `pil`, you gived {engine}. Skipping.')
