import argparse
import ast
import logging
import os
from typing import Optional

from billys.pipeline import train_pipeline, classify_pipeline, classify_deterministic_pipeline, train_deterministic_pipeline, classify_2_pipeline
from billys.util import get_log_level


def parse_steps(args):
    """
    Parse the value for `steps` from `args` and return it if
    it is valid. If `steps` is None, return a default value.
    """
    steps = args.steps

    if steps is None:
        return get_steps()

    if type(steps) is not list:
        logging.error(
            f'The given steps {steps} should be of type list, using defaults.')
        return get_steps()

    return steps


def parse_config(args):
    """
    Parse the value for `config` from `args` and return it if
    it is valid. If `config` is None, return a default value.
    """
    config = args.config

    if config is not None:
        try:
            if os.path.isfile(config):
                logging.debug(f'Opening config file {config}')
                with open(config, 'r') as f:
                    return ast.literal_eval(f.read())
            else:
                logging.debug(f'Reading configurations from arg')
                return ast.literal_eval(config)
        except IOError as e:
            logging.error(f'Configuration reading failed, using defaults')
            logging.error(e)
            raise e
    else:
        return {}


# def parse_preset(args) -> Optional[PresetConfig]:
#     stage = args.preset

#     if stage is None:
#         return None

#     return PresetConfig(stage)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument('stage', type=str, default=None, choices=['train', 'classify', 'train-bow', 'classify-bow'],
                        nargs='?', help='Do the training from dataset')

    parser.add_argument('--config', metavar='config', type=str, default=None,
                        nargs='?', help='Path to a file with configs or a dict with configs themselves')

    # TODO: step management.
    # parser.add_argument('--steps', metavar='steps', type=str, default=None,
    #                     nargs='*', help='Pipeline steps')

    parser.add_argument('--log-level', metavar='level', type=str,
                        choices=['critical', 'error',
                                 'warning', 'info', 'debug'],
                        default='info',
                        nargs='?', help='Logging level. Use `debug` for verbose output.')

    args = parser.parse_args()

    # Set immediatly the log level.
    logging.basicConfig(level=get_log_level(args.log_level))
    logging.debug(args)

    # Args parsing
    stage = args.stage
    config = parse_config(args)

    if stage is not None:
        if stage == 'train':
            train_pipeline(**config)
        elif stage == 'classify':
            classify_2_pipeline(**config)
        elif stage == 'train-bow':
            train_deterministic_pipeline(**config)
        elif stage == 'classify-bow':
            classify_deterministic_pipeline(**config)
        else:
            raise ValueError(f'The stage {stage} is not valid.')
    else:
        logging.info(
            'Do nothing by default, run the program with `--help` to see options.')
