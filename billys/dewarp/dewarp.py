import numpy as np
# from google.colab.patches import cv2_imshow
import os
import cv2
import tensorflow as tf

from billys.dewarp.homography import HomographyDL


def warp_image(img, pts_src, pts_dst, grayscale=False):
    """
    Dewarp image

    :param img:
    :param pts_src:
    :param pts_dst:
    :return:
    """
    if grayscale:
        height, width = img.shape
    else:
        height, width, _ = img.shape

    #  Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    #  Warp source image to destination based on homography; format (4,2)
    return cv2.warpPerspective(src=img, M=h, dsize=(width, height))


def visualize_xy(X, Y):
    """
    Draw detected corner points on original image

    :param X:
    :param Y:
    :return:
    """
    # unpack corner points
    x1, y1, x2, y2, x3, y3, x4, y4 = Y
    img = X.copy()

    # draw circles
    img = cv2.circle(img, (int(x1), int(y1)), 5, (0, 0, 255))
    img = cv2.circle(img, (int(x2), int(y2)), 5, (0, 0, 255))
    img = cv2.circle(img, (int(x3), int(y3)), 5, (0, 0, 255))
    img = cv2.circle(img, (int(x4), int(y4)), 5, (0, 0, 255))
    return img


def scale_estim_corners(corners, scale_x, scale_y):
    """
    scale estimated corners to original image size

    :param corners:
    :param scale_x:
    :param scale_y:
    :return:
    """
    erg = np.zeros((4, 2))

    for idx, corner_tuple in enumerate(corners):
        erg[idx] = corner_tuple[0]*scale_x, corner_tuple[1]*scale_y

    return erg


def make_model(model_path):
    return tf.keras.models.load_model(model_path)


def dewarp_image(image, homography_model, smart_doc=False, grayscale=False):
    """
    Dewarp the image `image` with model `homography_model`.

    Parameters
    ----------
    image: The image data
    homography_model: The model data
    smart_doc: Whther we need to manually rotate the image
    grayscale: Whther the image is in greyscale
    """

    # network spatial input shape
    input_shape = (384, 256)

    # create empty instance
    homography_dl = HomographyDL(
        input=None, output=None, architecture=None, model_fn=None, grayscale=None)

    # load image
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = cv2.imread(image_path, 0)-
    # else:
    #     img = cv2.imread(image_path)

    # manually rotate (should be automated)
    if smart_doc:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # save original size to compute scaling factor
    if grayscale:
        org_y, org_x = image.shape
    else:
        org_y, org_x, _ = image.shape

    fac_y, fac_x = org_y/input_shape[0], org_x/input_shape[1]

    # resize (just for recovering homography)
    img_homography = cv2.resize(image, (input_shape[1], input_shape[0]))

    # adjust dimension for network
    if grayscale:
        img_homography_net = np.reshape(
            img_homography, (1, input_shape[0], input_shape[1], 1))
    else:
        img_homography_net = np.reshape(
            img_homography, (1, input_shape[0], input_shape[1], 3))

    # normalize
    img_homography_norm = img_homography_net/255.0

    # estimate corner positions
    corners = homography_dl.predict_corners(
        homography_model, img_homography_norm)
    print(corners)

    # unwarp imgage (original size)
    pts_src = np.reshape(corners, (4, 2))
    pts_src = scale_estim_corners(pts_src, fac_x, fac_y)
    pts_dst = np.array([[0, 0], [org_x, 0], [org_x, org_y],
                        [0, org_y]], dtype='float32')

    dewarped_image = warp_image(image, pts_src, pts_dst, grayscale)

    return dewarped_image
    # cv2.imshow('test', dewarped_image)
    # cv2.resizeWindow('test', 512, 512)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # tesseract
    # ocr.run_image_to_text_save(dewarped_image, os.path.splitext(img_nm)[0])
