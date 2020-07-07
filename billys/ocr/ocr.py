import cv2
import pytesseract


def ocr_data(image):
    """
    Perform the ocr on the image `image` and return a dict with ocr results.

    Parameters
    ----------
    image
        The image to ocr.

    Returns
    -------
    ocr: dict
        A dict with ocr data extracted with pytesseract.
        The dict is composed by the following columns:
            'level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
            'left', 'top', 'width', 'height', 'conf', 'text'
    """

    # In order to know available configurations, run
    # `tesseract --help-extra` and look for the section OCR options.
    # Our configuration specifies
    # * the ocr engine mode (--oem) as 1, that is 'Neural nets LSTM engine only.'
    # * the page segmentation mode (--psm) as 3, that is 'Fully automatic page segmentation, but no OSD. (Default)'
    custom_config = r'--oem 1 --psm 3 -l ita'

    return pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config=custom_config)
