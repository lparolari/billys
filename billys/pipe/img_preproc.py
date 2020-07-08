"""
Image preprocessing pipeline steps
"""

import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageOps
import piexif

from billys.dataset import read_image, save_image
from billys.dewarp.dewarp import dewarp_image, make_model
from billys.util import make_filename, ensure_dir


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
    df_out = df[[column for column in df.columns if column not in ['filename', 'is_pdf', 'is_good']]].copy()

    homography_model = make_model(homography_model_path)
    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        grayscale = row['grayscale']
        is_pdf = row['is_pdf']
        is_good = row['is_good']
        target_name = row['target_name']
        imdata = read_image(filename, is_pdf=is_pdf)

        if not is_good:
            # Dewarp the image only if it is bad.
            dewarped_imdata = dewarp_image(
                imdata, homography_model, grayscale=grayscale)
        else:
            dewarped_imdata = imdata
        
        new_filename = make_filename(filename=filename, step='dewarp', cat=target_name)
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
    -------
    df
        A new dataframe with follwing changes
         * 'filename', overwrited with new dewarped filenames
    """
    df_out = df[[column for column in df.columns if column not in ['filename']]].copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        target_name = row['target_name']

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
                    img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)

        new_filename = make_filename(filename=filename, step='rotation', cat=target_name)
        new_filename_list.append(new_filename)

        ensure_dir(new_filename)
        save_image(new_filename, img, dpi=(300,300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out


def brightness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance image brightness.
    Images are saved in the working directory under 'brightness' folder.

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
    df_out = df[[column for column in df.columns if column not in ['filename']]].copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        target_name = row['target_name']

        img = Image.open(filename)

        # brightness

        img = ImageEnhance.Brightness(img)
        
        brightness = 2.0 # increase brightness

        img = img.enhance(brightness)

        new_filename = make_filename(filename=filename, step='brightness', cat=target_name)
        new_filename_list.append(new_filename)

        ensure_dir(new_filename)
        save_image(new_filename, img, dpi=(300,300), engine='pil')

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
    df_out = df[[column for column in df.columns if column not in ['filename']]].copy()

    new_filename_list = []

    for index, row in df.iterrows():
        filename = row['filename']
        target_name = row['target_name']

        img = Image.open(filename)

        # contrast

        img = ImageEnhance.Contrast(img)

        contrast = 2.0 # increase contrast
        
        img = img.enhance(contrast)

        new_filename = make_filename(filename=filename, step='contrast', cat=target_name)
        new_filename_list.append(new_filename)

        ensure_dir(new_filename)
        save_image(new_filename, img, dpi=(300,300), engine='pil')

    df_out['filename'] = new_filename_list

    return df_out
