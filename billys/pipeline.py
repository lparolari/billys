import os
from enum import Enum
import typing

import pandas as pd
import cv2

from billys.dataset import fetch_billys, make_dataframe
from billys.dewarp.dewarp import dewarp_image, make_model
from billys.ocr.ocr import ocr_data
from billys.checkpoint import save, revert
from billys.util import get_data_home


def pipeline(data_home: str = os.path.join(os.getcwd(), 'dataset'),
             homography_model_path: str = os.path.join(os.getcwd(), 'resource', 'model', 'xception_10000.h5'),
             force_good: bool = True):
    """
    Run the training pipeline.

    Parameters
    ----------
    TODO
    """

    # Plese use only **kebab-cased** idenfiers for pipeline steps
    # (https://it.wikipedia.org/wiki/Kebab_case)
    # Other types of case could compromise the saving and loading
    # functions for checkpoints.

    steps = [
        ('fetch-billys', lambda *_: fetch(data_home)),
        ('init-dataframe', lambda dataset: init(dataset, force_good)),
        ('print', show),
        ('dewarp', lambda df: dewarp(df, homography_model_path)),
        ('contrast', contrast),
        ('ocr', ocr),
        ('show-boxed-text', show_boxed_text),
        ('dump-ocr', dump),
        ('print', show),
        ('feat-preproc', skip),
        ('train-classifier', skip)
    ]

    out = None

    for item in steps:
        step, func = item
        print(f'Performing step {step} ... ')

        prev_out = out
        out = func(prev_out)

    print('Pipeline completed.')

    return out


def fetch(data_home: typing.Optional[str] = None):
    """
    Fetch the dataset from the path with logic in :func:`billys.util.get_data_home` and
    return it.

    Parameters
    ----------
    data_home: default: None
        The directory from which retrieve the dataset. See :func:`billys.util.get_data_home`.

    Returns
    -------
    dataset
        The dataset, see :func:`billys.dataset.fetch_billys`.
    """
    return fetch_billys(data_home=data_home)


def init(dataset, force_good: bool = False) -> pd.DataFrame:
    """
    Initialize the dataframe from given dataset.

    Parameters
    ----------
    dataset: required
        The dataset loaded with :func:`billys.dataset.fetch_billys`.

    force_good: default: False
        Force all the samples in the dataframe to be marked as good and skip
        some pipeline steps like dewarping and contrast aumentation.

    Returns
    -------
    df
        A new dataframe built with :func:`billys.dataset.make_dataframe`.
    """
    return make_dataframe(dataset=dataset, force_good=force_good)


def dewarp(df: pd.DataFrame, homography_model_path: str) -> pd.DataFrame:
    """
    Foreach sample in the dataframe `df` use the model located at `homography_model_path`
    to dewarp the images. Dewarp only images that are "not good".

    Parameters
    ----------
    df
        The dataset as a dataframe.

    homography_model_path
        The path to the homography model file in `.h5` format. 

    Returns
    -------
    df
        A new dataframe where the column `data`is overwrited with the dewarped images data.
    """
    df_out = df[[column for column in df.columns if column != 'data']].copy()

    homography_model = make_model(homography_model_path)
    dewarped_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        imdata = row['data']
        grayscale = row['grayscale']
        smart_doc = row['smart_doc']
        good = row['good']

        if not good:
            # Dewarp the image only if it is bad.
            dewarped_imdata = dewarp_image(
                imdata, homography_model, grayscale=grayscale, smart_doc=smart_doc)
            dewarped_list.append(dewarped_imdata)
        else:
            dewarped_list.append(imdata)

    df_out['data'] = dewarped_list

    return df_out


def contrast(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: docs and implementation
    # The identity function, for now.
    return df


def ocr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the ocr on images data.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    ------- 
    df
        A new dataframe with a new column `ocr` that contains a dict
        with extracted features from image. The type of every value
        in this column is documented at :func:`billys.ocr.ocr.ocr_data`.
    """
    df_out = df.copy()

    dict_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        imdata = row['data']

        ocr_dict = ocr_data(imdata)
        dict_list.append(ocr_dict)

    df_out['ocr'] = dict_list

    return df_out


def show(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print the dataframe as a side effect and return it.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        The dataframe itself without changes.
    """
    print(df)
    return df


def skip(x):
    """
    The identity function.
    """
    return x


def dump(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dump the dataframe on file as a side effect with :func:`billys.checkpoint.save`,
    and returns the dataframe without changes.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        The dataframe without changes.
    """
    filename = save('dump_ocr', df)
    print(f'Dumped object into {filename}')
    return df


def show_boxed_text(df: pd.DataFrame):
    """
    Save iamges with boxed words as a side effect and return the
    dataframe without changes. Images are saved in the `boxed`
    directory inside the path returned by :func:`get_data_home`.

    Parameters
    ----------
    df
        The dataset as a dataframe.
        Requires the columns
         * 'data', the image with contrast and illumination;
         * 'ocr', the dict with ocr features;
         * 'filename', the original image filename.

    Returns
    -------
    df
        The dataframe without changes.
    """

    boxed_images_path = os.path.join(get_data_home(), 'boxed')
    os.makedirs(boxed_images_path, exist_ok=True)

    for index, row in df.iterrows():
        ocr_dict = row['ocr']
        imdata = row['data']
        filename = row['filename']

        n_boxes = len(ocr_dict['text'])
        for i in range(n_boxes):
            if int(ocr_dict['conf'][i]) > 60:
                (x, y, w, h) = (ocr_dict['left'][i], ocr_dict['top']
                                [i], ocr_dict['width'][i], ocr_dict['height'][i])
                imdata = cv2.rectangle(
                    imdata, (x, y), (x + w, y + h), (0, 255, 0), 2)

        imS = cv2.resize(imdata, (500, 700))

        name = os.path.basename(filename).split('.')[0] + '.jpg'
        new_filename = os.path.join(boxed_images_path, name)

        cv2.imwrite(new_filename, imS)

    return df
