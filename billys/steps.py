"""
Common and shared pipeline steps.
"""

from billys.util import ensure_dir, get_data_home, make_dataset_filename, read_file
import logging
import os
from typing import Optional, Any, List

import cv2
import pandas as pd
import piexif
from PIL import Image, ImageEnhance, ImageOps

from billys.dataset import fetch_billys, make_dataframe_from_dataset, make_dataframe_from_filenames
from billys.dewarp.dewarp import dewarp_image, make_model
from billys.ocr.ocr import ocr_data
from billys.text.preprocessing import preprocess, make_nlp, download_stopwords
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


def fetch_dataset(data_home: Optional[str] = None, name: str = 'billys', subset: str = 'train'):
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


def fetch_filenames(filenames, data_home=None) -> List[str]:
    """
    Fetch images filenames.

    Parameters
    ----------
    filenames
            A list of image names
        or, a single image file name
        or, a directory name

        Note: the image path must be specified with the `data_home` param.

    data_home
        The path to data.

    Returns
    -------
    filenames
        A list of filenames.
    """
    images = filenames
    if type(images) is list:
        # it is already a list of filenames,
        # just join them with data_home
        return [get_filename(image, data_home=data_home) for image in images]

    if type(images) is str:
        if os.path.isfile(images):
            # it is a single filename
            return [get_filename(images, data_home=data_home)]
        else:
            # it is a directory, get the list of files
            return [get_filename(image, data_home=data_home) for image in images]


def fetch_data_and_classifier(dataset: str, classifier: str, data_home: str = None):
    return {
        'dataset': revert(dataset, data_home=data_home),
        'classifier': revert(classifier, data_home=data_home)
    }


def build_dataframe_from_dataset(dataset, force_good: bool = False, subset: str = 'train') -> pd.DataFrame:
    """
    Initialize the dataframe from given dataset. The dataset
    can be used for the training phase.

    Parameters
    ----------
    dataset
        The dataset loaded with :func:`billys.dataset.fetch_billys`.

    subset
        The subset of loaded dataset. Can be one of 'train', 'test'.

    force_good
        Force all the samples in the dataframe to be marked as good and skip
        some pipeline steps like dewarping and contrast aumentation.

    Returns
    -------
    df
        A new dataframe built with :func:`billys.dataset.make_dataframe_from_dataset`.
    """
    return make_dataframe_from_dataset(dataset=dataset, force_good=force_good, subset=subset)


def build_dataframe_from_filenames(filenames: List[str], force_good: bool = False) -> pd.DataFrame:
    """
    Initialize the dataframe from given filenames. The dataset
    can be used only for the classification task.

    Parameters
    ----------
    filenames
        A list of filenames loaded with :func:`billys.dataset.fetch_filenames`.

    force_good
        Force all the samples in the dataframe to be marked as good and skip
        some pipeline steps like dewarping and contrast aumentation.

    Returns
    -------
    df
        A new dataframe built with :func:`billys.dataset.make_dataframe_from_filenames`.
    """
    return make_dataframe_from_filenames(filenames=filenames, force_good=force_good)


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
        The dataset as a dataframe.

    homography_model_path
        The path to the homography model file in `.h5` format.

    Returns
    -------
    df
        A new dataframe with follwing changes
         * 'filename', overwrited with new dewarped filenames
    """
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
            copyfile(filename, new_filename)

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
    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']

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

        new_filename = make_filename(row, step='rotation')
        new_filename_list.append(new_filename)

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
    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']

        img = Image.open(filename)

        logging.debug(f'Brightening image {filename}')

        # brightness

        img = ImageEnhance.Brightness(img)

        brightness = 2.0  # increase brightness

        img = img.enhance(brightness)

        new_filename = make_filename(row, step='brightness')
        new_filename_list.append(new_filename)

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
    df_out = df.copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']

        img = Image.open(filename)

        logging.debug(f'Contrasting image {filename}')

        # contrast

        img = ImageEnhance.Contrast(img)

        contrast = 2.0  # increase contrast

        img = img.enhance(contrast)

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

        text = preprocess(text=text, nlp=nlp, use_lemmatize=False)

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


def classify(data):
    dataset, classifier = data['dataset'], data['classifier']

    print(dataset)
    print(classifier)
    print(classifier.predict(dataset))
