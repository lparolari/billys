import cv2
import pytesseract


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def ocr_data(image):
    image = get_grayscale(image)
    image = thresholding(image)

    # Adding custom options
    custom_config = r'-c tessedit_char_whitelist=0123 --psm 6 -l ita'
    #text = pytesseract.image_to_string(img, config=custom_config)
    # print(text)

    d = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config=custom_config)
    # dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])

    return d


# def ocr_text(image):
#     image = get_grayscale(image)
#     image = thresholding(image)

#     # Adding custom options
#     custom_config = r'-c tessedit_char_whitelist=0123 --psm 6 -l ita'
#     #text = pytesseract.image_to_string(img, config=custom_config)
#     # print(text)

#     d = pytesseract.image_to_string(
#         image, output_type=pytesseract.Output.DICT, config=custom_config)

#     return d['text']
