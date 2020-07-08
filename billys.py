import argparse
import logging

from billys.pipeline import pipeline, make_steps, make_default_config

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Process some integers.')

    # parser.add_argument('--use-checkpoints',
    #                     metavar='use_checkpoints',
    #                     type=bool,
    #                     nargs='?',
    #                     default=True,
    #                     help='''enable or disable checkpoints. Note that if you disable checkpoints,
    #                             your pipeline should start from step 0.''')

    args = parser.parse_args()

    pipeline(make_steps([
        'fetch-billys',
        'init-dataframe',
        'print',
        'dewarp',
        'rotation',
        'brightness',
        'fetch-checkpoint',
        'ocr',
        'show-boxed-text',
    ], {**make_default_config()}))
