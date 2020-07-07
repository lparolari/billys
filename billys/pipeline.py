import os
from enum import Enum

import pandas as pd

from billys.dataset import fetch_billys, make_dataframe
from billys.dewarp.dewarp import dewarp_image, make_model
from billys.ocr.ocr import ocr_data
from billys.checkpoint import save, revert


def pipeline():
    """
    Run the training pipeline starting from the step `step`.

    Parameters
    ----------
    TODO
    """

    # Plese use only kebab-cased idenfiers for pipeline steps
    # (https://it.wikipedia.org/wiki/Kebab_case)

    data_home = os.path.join(os.getcwd(), 'dataset')
    homography_model_path = os.path.join(
        os.getcwd(), 'resource', 'model', 'xception_10000.h5')
    force_good = True

    steps = [
        ('init', lambda *_: init(data_home, force_good)),
        ('print', show),
        ('dewarp', lambda df: dewarp(df, homography_model_path)),
        ('contrast', contrast),
        ('ocr', ocr),
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


def show(df: pd.DataFrame) -> pd.DataFrame:
    print(df)
    return df


def init(data_home: str, force_good: bool = False) -> pd.DataFrame:
    return make_dataframe(
        fetch_billys(data_home=data_home), force_good=force_good)


def dewarp(df: pd.DataFrame, homography_model_path: str) -> pd.DataFrame:
    """
    Return a new dataframe where column `data` now contains the dewarped images data.
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
    return df


def ocr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a new dataframe where the column `data` contains 
    the text extracted from images.
    """
    df_out = df[[column for column in df.columns if column != 'data']].copy()

    dict_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        imdata = row['data']

        print('Processing image {} ...'.format(filename))

        ocr_dict = ocr_data(imdata)

        dict_list.append(ocr_dict)

    df_out['data'] = dict_list

    return df_out


def skip(x):
    return x


def dump(df):
    filename = save('dump_ocr', df)
    print(f'Dumped object into {filename}')
    return df


class Step(Enum):
    INIT = 0
    DEWARP = 1
    CONTRAST = 2
    OCR = 3
    FEAT_PREPROC = 4
    TRAIN_CLASSIFIER = 5

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not ((self < other) and (self == other))

    def __ge__(self, other):
        return not (self < other)

    def __ne__(self, other):
        return not (self == other)

    def __int__(self):
        return self.value

    def __hash__(self):
        return self.value
