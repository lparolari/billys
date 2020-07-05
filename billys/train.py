# 1. Dataset collection. We need to gather the dataset and unzip it to a folder.
# 2. Images collection. We need to create a structure (like a pandas dataframe) containing, for each image, its path, the image content read with `cv2.imread`, and the label.
# 3. Images preprocessing. Images needs to be transformed in order to perform the ocr. As a transformation we could
#   - augment the contrast and the illumination of the image
#   - rotate and crop the image (see reference [1]).
# 4. Features extraction. We need to extract features such as words from the images. This words can be extracted through the OCR by means of tools like Tesseract (see reference [2]).
# 5. Feature preprocessing. As features will be natural language words, we need to preprocess extracted text and remove non relevant words or too frequent words.
# 6. Image classification. We can elaborate the extracted feature in order to classify a bill's image into its class.

import os
from enum import Enum
from billys.dataset import fetch_billys, save_checkpoint, load_checkpoint


def dewarp(dataset):
    raise NotImplementedError


def augment_contrast(dataset):
    raise NotImplementedError


def ocr(dataset):
    raise NotImplementedError


def feat_preproc(dataste):
    raise NotImplementedError


def train_classifier(dataset):
    raise NotImplementedError


def pipeline(step: int = 0):
    """
    Run the training pipeline starting from the step `step`.

    Parameters
    ----------
    step: int, default: 0
        The starting point for the training pipeline.
        Can be one in the `Step` enum.
    """

    # Phase A: load dataset
    if Step(step) <= Step.INIT:
        data_home = os.path.join(os.getcwd(), 'dataset')
        train = fetch_billys(data_home=data_home)
        save_checkpoint(Step.INIT, train)

    # Phase B: preprocess images
    if Step(step) <= Step.DEWARP:
        checkpoint = load_checkpoint(Step.DEWARP)
        dewarped = dewarp(train)
        save_checkpoint(Step.DEWARP, dewarped)

    if Step(step) <= Step.CONTRAST:
        checkpoint = load_checkpoint(Step.CONTRAST)
        contrasted = augment_contrast(checkpoint)
        save_checkpoint(Step.CONTRAST, contrasted)

    # Phase C: feature extraction (OCR)
    if Step(step) <= Step.ORC:
        checkpoint = load_checkpoint(Step.ORC)
        text = ocr(checkpoint)
        save_checkpoint(Step.ORC, text)

    # Phase D: feature preprocessing
    if Step(step) <= Step.FEAT_PREPROC:
        checkpoint = load_checkpoint(Step.FEAT_PREPROC)
        text_preproc = feat_preproc(checkpoint)
        save_checkpoint(Step.FEAT_PREPROC, text_preproc)

    # Phase E: classification
    if Step(step) <= Step.TRAIN_CLASSIFIER:
        checkpoint = load_checkpoint(Step.TRAIN_CLASSIFIER)
        classifier = train_classifier(checkpoint)

    # TODO: save the trained model

    return classifier


class Step(Enum):
    INIT = 0
    DEWARP = 1
    CONTRAST = 2
    ORC = 3
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
