import argparse
import logging

from billys.pipeline import pipeline

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

    pipeline()
