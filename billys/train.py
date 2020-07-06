# 1. Dataset collection. We need to gather the dataset and unzip it to a folder.
# 2. Images collection. We need to create a structure (like a pandas dataframe) containing, for each image, its path, the image content read with `cv2.imread`, and the label.
# 3. Images preprocessing. Images needs to be transformed in order to perform the ocr. As a transformation we could
#   - augment the contrast and the illumination of the image
#   - rotate and crop the image (see reference [1]).
# 4. Features extraction. We need to extract features such as words from the images. This words can be extracted through the OCR by means of tools like Tesseract (see reference [2]).
# 5. Feature preprocessing. As features will be natural language words, we need to preprocess extracted text and remove non relevant words or too frequent words.
# 6. Image classification. We can elaborate the extracted feature in order to classify a bill's image into its class.

import os

from billys.dataset import fetch_billys
from billys.checkpoint import save, revert
from billys.pipeline import Step
from billys.dewarp.dewarp import dewarp
from billys.ocr.ocr import ocr


def dewarp_step(dataset):
    return dataset


def augment_contrast(dataset):
    return dataset


def ocr_step(dataset):
    for image_path in dataset.filenames:
        print(ocr(image_path))


def feat_preproc(dataset):
    return dataset


def train_classifier(dataset):
    return dataset


def pipeline(first_step: Step = Step.INIT, use_checkpoints: bool = False):
    """
    Run the training pipeline starting from the step `step`.

    Parameters
    ----------
    step: int, default: 0
        The starting point for the training pipeline.
        Can be one in the `Step` enum.
    """
    data_home = os.path.join(os.getcwd(), 'dataset')
    steps = {
        Step.INIT: (lambda: fetch_billys(data_home=data_home)),
        Step.DEWARP: (lambda checkpoint: dewarp_step(checkpoint)),
        Step.CONTRAST: (lambda checkpoint: augment_contrast(checkpoint)),
        Step.OCR: (lambda checkpoint: ocr_step(checkpoint)),
        Step.FEAT_PREPROC: (lambda checkpoint: feat_preproc(checkpoint)),
        Step.TRAIN_CLASSIFIER: (lambda checkpoint: train_classifier(checkpoint)),
    }

    for step, func in steps.items():
        print(f'Performing step {int(step)} ... ')
        if first_step <= step:
            if step is Step.INIT:
                checkpoint = func()
                if use_checkpoints is True:
                    save(step, checkpoint)
            else:
                if use_checkpoints is True:
                    checkpoint = revert(step)
                checkpoint = func(checkpoint)
                if use_checkpoints is True:
                    save(step, checkpoint)

    print('Pipeline completed.')

    classifier = revert(Step.TRAIN_CLASSIFIER)

    return classifier
