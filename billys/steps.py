"""
Common and shared pipeline steps.
"""

from billys.util import ensure_dir, get_data_home, make_dataset_filename, read_file
import logging
import os
from typing import Optional, Any

import cv2
import pandas as pd
import piexif
from PIL import Image, ImageEnhance, ImageOps

from billys.dataset import fetch_billys, make_dataframe
from billys.dewarp.dewarp import dewarp_image, make_model
from billys.ocr.ocr import ocr_data
from billys.text.preprocessing import (to_lower, remove_accented_chars, remove_punctuation,
                                       remove_nums, remove_stopwords, lemmatize, tokenize,
                                       download_stopwords, make_nlp)
from billys.text.classification import train
from billys.util import get_filename, read_file, save_file, read_dump, save_dump, read_image, save_image


"""
Shared pipeline steps.
"""


def show(obj: Any) -> Any:
    """
    Print the dataframe as a side effect and return it.
    Works only if logging level is setted to debug.

    Parameters
    ----------
    obj
        The object to log.

    Returns
    -------
    df
        The dataframe itself without changes.
    """
    logging.debug(obj)
    return obj


def skip(obj: Any) -> Any:
    """
    The identity function.
    """
    return obj


def dump(obj, name: str, data_home: Optional[str] = None):
    """
    Dump given object on a file and return the object itself.

    Parameters
    ----------
    obj
        The dataset as a dataframe.

    name
        The dump name, e.g. 'mydump'.

    data_home
        Where to save the object dump.

    Returns
    -------
    obj
        The object without changes.
    """
    filename = get_filename(name, data_home=data_home)
    save_dump(obj=obj, filename=filename)
    logging.info(f'Dumped object to {filename}')
    return obj


def revert(name: str, data_home: Optional[str] = None) -> Any:
    """
    Fetch the dumped object and return it.

    Parameters
    ----------
    name
        The dump name, e.g. 'mydump'.

    data_home
        Where to save the object dump.

    Returns
    -------
    obj
        The object dumped object.
    """
    return read_dump(filename=get_filename(name, data_home=data_home))


"""
Dataset initialization pipeline steps
"""


def fetch(data_home: Optional[str] = None, name: str = 'billys', subset: str = 'train'):
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


"""
Image preprocessing pipeline steps
"""


def dewarp(df: pd.DataFrame, homography_model_path: str) -> pd.DataFrame:
    """
    Foreach sample in the dataframe `df` use the model located at `homography_model_path`
    to dewarp the images. Dewarp only images that are "not good".
    Dewarped images are saved in the working directory under 'dewarp' folder.

    Parameters
    ----------
    df
        The dataset as a dataframe. Required columns are
            'filename', 'grayscale', 'is_good'

    homography_model_path
        The path to the homography model file in `.h5` format.

    Returns
    -------
    df
        A new dataframe with follwing changes
         * 'is_pdf', dropped
         * 'is_good', dropped
         * 'filename', overwrited with new dewarped filenames
    """
    df_out = df[[column for column in df.columns if column not in [
        'filename', 'is_pdf', 'is_good']]].copy()

    homography_model = make_model(homography_model_path)
    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        grayscale = row['grayscale']
        is_pdf = row['is_pdf']
        is_good = row['is_good']
        target_name = row['target_name']
        subset = row['subset']
        imdata = read_image(filename, is_pdf=is_pdf)

        if not is_good:
            # Dewarp the image only if it is bad.
            logging.debug(f'Dewarping image {filename}')
            dewarped_imdata = dewarp_image(
                imdata, homography_model, grayscale=grayscale)
        else:
            logging.debug(f'Skipping dewarp for {filename}')
            dewarped_imdata = imdata

        new_filename = make_dataset_filename(
            filename=filename, step='dewarp', subset=subset, cat=target_name)
        new_filename_list.append(new_filename)

        save_image(new_filename, dewarped_imdata)

    df_out['filename'] = new_filename_list

    return df_out


def rotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct right image orientation.
    Images are saved in the working directory under 'rotation' folder.

    Parameters
    ----------
    df
        The dataset as a dataframe. Required columns are
            'filename', 'target_name'

    Returns
    - ------
    df
        A new dataframe with follwing changes
         * 'filename', overwrited with new dewarped filenames
    """
    df_out = df[[column for column in df.columns if column not in [
        'filename']]].copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        subset = row['subset']
        target_name = row['target_name']

        img = Image.open(filename)

        logging.debug(f'Rotating image {filename}')

        # load the image oriented corrected

        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])

            if piexif.ImageIFD.Orientation in exif_dict["0th"]:
                orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)

                if orientation == 2:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img = img.rotate(-90,
                                     expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(
                        Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)

        new_filename = make_dataset_filename(
            filename=filename, step='rotation', subset=subset, cat=target_name)
        new_filename_list.append(new_filename)

        ensure_dir(new_filename)
        save_image(new_filename, img, dpi=(300, 300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out


def brightness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance image brightness.
    Images are saved in the working directory under 'brightness' folder.

    Parameters
    - ---------
    df
        The dataset as a dataframe. Required columns are
            'filename', 'target_name'

    Returns
    -------
    df
        A new dataframe with follwing changes
         * 'filename', overwrited with new dewarped filenames
    """
    df_out = df[[column for column in df.columns if column not in [
        'filename']]].copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        subset = row['subset']
        target_name = row['target_name']

        img = Image.open(filename)

        logging.debug(f'Brightening image {filename}')

        # brightness

        img = ImageEnhance.Brightness(img)

        brightness = 2.0  # increase brightness

        img = img.enhance(brightness)

        new_filename = make_dataset_filename(
            filename=filename, step='brightness', subset=subset, cat=target_name)
        new_filename_list.append(new_filename)

        ensure_dir(new_filename)
        save_image(new_filename, img, dpi=(300, 300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out


def contrast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance image contrast.
    Images are saved in the working directory under 'contrast' folder.

    Parameters
    ----------
    df
        The dataset as a dataframe. Required columns are
            'filename', 'target_name'

    Returns
    -------
    df
        A new dataframe with follwing changes
         * 'filename', overwrited with new dewarped filenames
    """
    df_out = df[[column for column in df.columns if column not in [
        'filename']]].copy()

    print(df_out)

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        subset = row['subset']
        target_name = row['target_name']

        img = Image.open(filename)

        logging.debug(f'Contrasting image {filename}')

        # contrast

        img = ImageEnhance.Contrast(img)

        contrast = 2.0  # increase contrast

        img = img.enhance(contrast)

        new_filename = make_dataset_filename(
            filename=filename, step='contrast', subset=subset, cat=target_name)
        new_filename_list.append(new_filename)

        ensure_dir(new_filename)
        save_image(new_filename, img, dpi=(300, 300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out


"""
Ocr pipeline steps.
"""


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
        imdata = read_image(filename, is_pdf=False)

        logging.debug(f'Performing ocr on image {filename}')

        ocr_dict = ocr_data(imdata)
        dict_list.append(ocr_dict)

    df_out['ocr'] = dict_list

    return df_out


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
        filename = row['filename']
        target_name = row['target_name']
        subset = row['subset']
        imdata = read_image(filename, is_pdf=False)

        logging.debug(f'Boxing image {filename}')

        n_boxes = len(ocr_dict['text'])
        for i in range(n_boxes):
            if int(ocr_dict['conf'][i]) > 60:
                (x, y, w, h) = (ocr_dict['left'][i], ocr_dict['top']
                                [i], ocr_dict['width'][i], ocr_dict['height'][i])
                imdata = cv2.rectangle(
                    imdata, (x, y), (x + w, y + h), (0, 255, 0), 2)

        img = cv2.resize(imdata, (500, 700))

        new_filename = make_dataset_filename(
            filename=filename, step='boxed', subset=subset, cat=target_name)

        ensure_dir(new_filename)
        save_image(new_filename, img)

    return df


"""
Text preprocessing steps
"""


def extract_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO
    """

    df_out = df.copy()

    text_list = []

    for index, row in df.iterrows():
        ocr_dict = row['ocr']
        filename = row['filename']

        logging.debug(f'Extracting text for {filename}')

        text = " ".join(ocr_dict['text'])

        text_list.append(text)

    df_out['text'] = text_list

    return df_out


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO
    """

    df_out = df.copy()

    download_stopwords()
    nlp = make_nlp()

    text_list = []

    for index, row in df.iterrows():
        ocr_dict = row['ocr']
        filename = row['filename']
        text = row['text']

        logging.debug(f'Preprocessing text for {filename}')

        text = to_lower(text)
        text = remove_accented_chars(text)
        text = remove_punctuation(text)
        # text = lemmatize(text, nlp)
        text = remove_nums(text)
        text = remove_stopwords(text)

        text_list.append(text)

    df_out['text'] = text_list

    return df_out


"""
Text classification steps
"""


def train_classifier(data):
    """
    Train the classifier and returns it.
    The classifier specification is given in :func:`billys.text.classification.train`.

    Parameters
    ----------
    data
        A tuple where the first component is a dataframe used by the training phase,
        while the second is is used for the test phase.

    Returns
    -------
    out
        A dict where with two keys
         * 'data', whose value is a pair (train_df, test_df)
         * 'classifier', whose value is a scikit-learn classifier
    """
    train_df, test_df = data

    X_train = train_df['text'].to_list()
    y_train = train_df['target'].to_list()
    X_test = test_df['text'].to_list()
    y_test = test_df['target'].to_list()

    clf = train(X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test)

    return {'data': data, 'classifier': clf}
