"""
The main module for pipeline steps.

This module contains all pipeline functions and step that a generic
pipeline can execute. Steps can be recombine together in order to
create new pipelines. For every step an input-output specification
is given, so step combination should be almost straighforward.

Steps are categorized in
* shared/common, can be used at any point of the pipeline
* init, can be used for dataset/files fetching and dataframe creation
* images processing, transform images and process them
* text processing, extract text features and process them
* classifier training, train the classifier
* classification, do the image classification part
"""

import logging
import os
from typing import Any, List, Optional, Union

import cv2
import pandas as pd
import piexif
from PIL import Image, ImageEnhance, ImageOps

from billys.dataset import (fetch_billys, make_dataframe_from_datasets,
                            make_dataframe_from_filenames, make_filename)
from billys.dewarp.dewarp import dewarp_image, make_model
from billys.ocr.ocr import ocr_data
from billys.text.classification import train
from billys.text.preprocessing import download_stopwords, make_nlp, preprocess
from billys.util import (ensure_dir, get_data_home, get_filename,
                         make_dataset_filename, read_dump, read_file,
                         read_image, save_dump, save_file, save_image)
from sklearn import metrics

"""
Shared pipeline steps.
"""


def log(obj: Any) -> Any:
    """
    Print the given object as a side effect and return it.
    Works only if logging level is setted to debug.

    Parameters
    ----------
    obj
        The object to log.

    Returns
    -------
    obj
        The given object without changes.
    """
    logging.info('Logging object ...')
    logging.debug(obj)
    return obj


def skip(obj: Any) -> Any:
    """
    The identity function.
    """
    logging.info('Skipping ...')
    return obj


def dump(obj: Any, name: str, data_home: Optional[str] = None) -> Any:
    """
    Dump given object on a file and return the object itself.

    Parameters
    ----------
    obj
        The object to dump.

    name
        The dump name, e.g. 'mydump.pkl'.

    data_home
        The path to data dir.

    Returns
    -------
    obj
        The object without changes.
    """
    logging.info('Dumping object ...')

    filename = get_filename(name, data_home=data_home)
    save_dump(obj=obj, filename=filename)

    # Informing the user for the dumped object.
    logging.info(f'Dumped object to {filename}')

    return obj


def revert(name: str, data_home: Optional[str] = None) -> Any:
    """
    Fetch a dumped object in pickle format and return it.

    Parameters
    ----------
    name
        The dump name, e.g. 'mydump.pkl'.

    data_home
        The path to data dir.

    Returns
    -------
    obj
        The dumped object.
    """
    logging.info('Reverting dump ...')
    return read_dump(filename=get_filename(name, data_home=data_home))


"""
Dataset initialization pipeline steps
"""


def fetch_dataset(name: str = 'billys', data_home: Optional[str] = None):
    """
    Fetch the whole dataset with name `name`.

    Parameters
    ----------
    name
        The dataset name.

    data_home
        The path to data dir.

    Returns
    -------
    dataset
        The dataset as a dict with keys `train` and `test`.
        For further details, see :func:`billys.dataset.fetch_billys`.
    """
    logging.info('Fetching dataset ...')
    return fetch_billys(data_home=data_home, name=name, subset='all')


def fetch_filenames(filenames: Union[List[str], str], data_home=None) -> List[str]:
    """
    Fetch images fully qualified filenames for given filenames or
    direcoty name.

    Parameters
    ----------
    filenames
            A list of image names
        or, a single image file name
        or, a directory name

        Note: the image path must be specified with the `data_home` param.

    data_home
        The path to data dir.

    Returns
    -------
    filenames
        A list of fully qualified filenames.
    """
    logging.info('Fetching filenames ...')

    if type(filenames) is list:
        # it is already a list of filenames,
        # just join them with data_home
        return [get_filename(filename, data_home=data_home) for filename in filenames]

    if type(filenames) is str:
        if os.path.isfile(filenames):
            # it is a single filename
            return [get_filename(filenames, data_home=data_home)]
        else:
            # it is a directory, get the list of files
            from os import listdir
            from os.path import isfile

            path = filenames

            fq_filenames = [get_filename(os.path.join(path, filename), data_home=data_home)
                            for filename in listdir(get_filename(path, data_home=data_home))]

            return [fn for fn in fq_filenames if isfile(fn)]


def build_dataframe(input_data, input_type: str, force_good: bool = False) -> pd.DataFrame:
    """
    Initialize the DataFrame for the pipeline from input, setting
    the stage `stage` and forcing images as good if `force_good`
    holds.

    Parameters
    ----------
    input_data
            A dataset dict with train and test datasets
        or, a list of filenames / filename / directory name.

    input_type
        The input type. Can be one of `datasets`, `filenames`.

    force_good
        Force all the samples in the dataframe to be marked as good,
        i.e. they don't need image preprocessing steps.

    Returns
    -------
    df
        A new dataframe built with input data.
        See :func:`billys.dataset.make_dataframe_from_datasets`
        and :func:`billys.dataset.make_dataframe_from_filenames`
        for more details on the dataframe columns.
    """
    logging.info(f'Building dataframe from {input_type} ...')

    if input_type == 'datasets':
        return make_dataframe_from_datasets(
            datasets=input_data,
            force_good=force_good)
    elif input_type == 'filenames':
        return make_dataframe_from_filenames(
            filenames=input_data,
            force_good=force_good)
    else:
        raise ValueError(
            f'Cannot build dataframe with input type {input_type}.')


def convert_to_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all pdf in `df` to images, and copy file that are
    already images without changes.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        A new dataframe with the following changes:
         * 'filename', overwrited with new converted filenames.
    """
    logging.info('Converting pdf to images ...')

    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        is_pdf = row['is_pdf']

        new_filename = make_filename(row, step='to_image', ext='jpg')
        new_filename_list.append(new_filename)

        if is_pdf:
            logging.debug(f'Converting image {filename}')
            logging.debug(f'New filename {new_filename}')

            # read image, even if it is a pdf, and write it as
            # a real image.
            imdata = read_image(filename, is_pdf=is_pdf)
            save_image(new_filename, imdata)
        else:
            logging.debug(f'Skipping conversion for {filename}, copying')
            from shutil import copyfile
            ensure_dir(new_filename)
            copyfile(filename, new_filename)

    df_out['filename'] = new_filename_list

    return df_out


"""
Image preprocessing pipeline steps
"""


def dewarp(df: pd.DataFrame, homography_model_path: str) -> pd.DataFrame:
    """
    Foreach sample in the dataframe `df` use the model located at
    `homography_model_path` to dewarp the images. Dewarp only images
    that are "not good", i.e. df['is_good`] == False.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    homography_model_path
        The path to the homography model file in `.h5` format.

    Returns
    -------
    df
        A new dataframe with follwing changes:
         * 'filename', overwrited with new dewarped filenames.
    """
    logging.info('Dewarping images ...')

    df_out = df.copy()

    homography_model = make_model(homography_model_path)
    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        is_good = row['is_good']

        new_filename = make_filename(row, step='dewarp')
        new_filename_list.append(new_filename)

        if not is_good:
            # Dewarp the image only if it is bad.
            logging.debug(f'Dewarping image {filename}')
            grayscale = row['is_grayscale']
            is_pdf = row['is_pdf']

            imdata = read_image(filename, is_pdf=is_pdf)  # read

            dewarped_imdata = dewarp_image(
                imdata, homography_model, grayscale=grayscale)  # dewarp

            save_image(new_filename, dewarped_imdata)  # save
        else:
            logging.debug(f'Skipping dewarp for {filename}')
            from shutil import copyfile
            ensure_dir(new_filename)
            copyfile(filename, new_filename)

    df_out['filename'] = new_filename_list

    return df_out


def rotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the right image orientation.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        A new dataframe with follwing changes:
         * 'filename', overwrited with new rotated filenames.
    """
    logging.info('Rotating images ...')

    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']

        logging.debug(f'Rotating image {filename}')

        img = Image.open(filename)

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

        new_filename = make_filename(row, step='rotation')
        new_filename_list.append(new_filename)

        save_image(new_filename, img, dpi=(300, 300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out


def brightness(df: pd.DataFrame, gain: float = 1.5) -> pd.DataFrame:
    """
    Enhance image brightness.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        A new dataframe with follwing changes:
         * 'filename', overwrited with new brightened image filenames.
    """
    logging.info('Brightening images ...')

    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']

        logging.debug(f'Brightening image {filename}')

        img = Image.open(filename)

        img = ImageEnhance.Brightness(img)
        img = img.enhance(gain)

        new_filename = make_filename(row, step='brightness')
        new_filename_list.append(new_filename)

        save_image(new_filename, img, dpi=(300, 300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out


def contrast(df: pd.DataFrame, gain: float = 1.5) -> pd.DataFrame:
    """
    Enhance image contrast.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        A new dataframe with follwing changes:
         * 'filename', overwrited with new contrasted image filenames.
    """
    logging.info('Contrasting images ...')

    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']

        img = Image.open(filename)

        logging.debug(f'Contrasting image {filename}')

        img = ImageEnhance.Contrast(img)
        img = img.enhance(gain)

        new_filename = make_filename(row, step='contrast')
        new_filename_list.append(new_filename)

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
        A new dataframe with the following changes:
        - `ocr`, new column; contains a dict with extracted text
           features from image. See :func:`billys.ocr.ocr.ocr_data`
           for further information on the type of this column.
    """
    logging.info('Performing ocr ...')

    df_out = df.copy()

    dict_list = []
    total_rows = df.shape[0]
    processed = 0

    for index, row in df.iterrows():
        processed = processed + 1
        filename = row['filename']
        imdata = read_image(filename, is_pdf=False)

        logging.debug(f'Performing ocr {processed} of {total_rows} on image {filename} ...')

        ocr_dict = ocr_data(imdata)
        dict_list.append(ocr_dict)

    df_out['ocr'] = dict_list

    return df_out


def show_boxed_text(df: pd.DataFrame):
    """
    Save iamges with boxed words as a side effect and return the
    dataframe without changes. Images are saved in the working dir
    under `boxed` directory.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        The dataframe without changes.
    """
    logging.info('Showing boxed images ...')

    boxed_images_path = os.path.join(get_data_home(), 'boxed')
    os.makedirs(boxed_images_path, exist_ok=True)

    for index, row in df.iterrows():
        ocr_dict = row['ocr']
        filename = row['filename']
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

        new_filename = make_filename(row, step='boxed')

        save_image(new_filename, img)

    return df


"""
Text preprocessing steps
"""


def extract_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text features from the ocr column, disregarding non
    relevant information.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        A new dataframe with the following changes:
         * 'text', new column; contains textual features extracted
           from each image.
    """
    logging.info('Extracting text features ...')

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
    Preprocess textual features with a common text preprocessing
    pipeline. For further details, see
        :func:`billys.text.preprocessing.preprocess`.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        A new dataframe with the following changes:
         * 'text', contains preprocessed text for each image.
    """
    logging.info('Preprocessing text ...')

    df_out = df.copy()

    download_stopwords()
    nlp = make_nlp()

    text_list = []

    for index, row in df.iterrows():
        ocr_dict = row['ocr']
        filename = row['filename']
        text = row['text']

        logging.debug(f'Preprocessing text for {filename}')

        text = preprocess(text=text, nlp=nlp, use_lemmatize=False)

        text_list.append(text)

    df_out['text'] = text_list

    return df_out


"""
Text classification steps
"""


def train_classifier(df: pd.DataFrame):
    """
    Train the classifier and returns it. The classifier specification
    is given in
        :func:`billys.text.classification.train`.

    Parameters
    ----------
    df
        The dataset as a dataframe. The dataframe should contains both
        `train` and `test` entries.

    Returns
    -------
    clf
        The trained classifier.
    """
    logging.info('Training classifier ...')

    train_df = df.loc[df['subset'] == 'train']
    test_df = df.loc[df['subset'] == 'test']

    X_train = train_df['text'].to_list()
    y_train = train_df['target'].to_list()
    X_test = test_df['text'].to_list()
    y_test = test_df['target'].to_list()

    clf = train(X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test)

    return clf


def classify(df: pd.DataFrame, clf):
    """
    Perform the classification task among `df` using the classifier
    `clf`, and return the predicted values.

    Parameters
    ----------
    df
        The dataset as a dataframe.
    clf
        An already trained scikit classifier.

    Returns
    -------
    predicted
        Predicted target values for entries in `df`.
    """
    logging.info('Classifying ...')

    texts = df['text'].tolist()
    return clf.predict(texts)


# def train_bow(df):
#     # TODO: docs

#     bag_of_words_per_category = []

#     for i in range(5):
#         cat = df[df['target'] == i]

#         data = cat['text']

#         from sklearn.feature_extraction.text import CountVectorizer

#         # use the scikit vectorized for creating the bag of words
#         vectorizer = CountVectorizer().fit(data)
#         bag_of_words = vectorizer.transform(data)

#         # create the sum of the bag of words in order to represent frequencies
#         sum_words = bag_of_words.sum(axis=0)

#         words_freq = [(word, sum_words[0, idx])
#                       for word, idx in vectorizer.vocabulary_.items()]
#         words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

#         # print(len(words_freq))
#         words_freq = words_freq[:(int(len(words_freq) * 0.1))]

#         # if f <= 10 and f >= 2]
#         words_freq = [(w, f) for (w, f) in words_freq if (
#             f <= 10 * len(cat) and f >= len(cat))]
#         print(words_freq)

#         bag_of_words_per_category.append(words_freq)

#     dump(bag_of_words_per_category, name='bag_of_words.pkl')
