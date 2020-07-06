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


def ocr_image(image):
    image = get_grayscale(image)
    image = thresholding(image)

    # Adding custom options
    custom_config = r'-c tessedit_char_whitelist=0123 --psm 6 -l ita'
    #text = pytesseract.image_to_string(img, config=custom_config)
    # print(text)

    d = pytesseract.image_to_string(
        image, output_type=pytesseract.Output.DICT, config=custom_config)

    return d['text']
