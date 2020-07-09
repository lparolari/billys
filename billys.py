import argparse
import ast
import logging

from billys.pipeline import pipeline, make_steps, make_config, get_default_steps
from billys.util import get_log_level


def parse_steps(args):
    return args.steps


def parse_config(args):
    config = args.config

    try:
        if os.path.isfile(config):
            with open(config, 'r') as f:
                return ast.literal_eval(f.read())
        else:
            return ast.literal_eval(config)
    except:
        return {}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--steps', metavar='steps', type=str, default=get_default_steps(),
                        nargs='*', help='Pipeline steps')

    parser.add_argument('--config', metavar='config', type=str, default=None,
                        nargs='?', help='Path to a file with configs or a dict with configs themselves')

    parser.add_argument('--log-level', metavar='log_level', type=str,
                        choices=['critical', 'error',
                                 'warning', 'info', 'debug'],
                        default='info',
                        nargs='?', help='Working directory')

    args = parser.parse_args()

    logging.basicConfig(level=get_log_level(args.log_level))
    logging.debug(args)

    # Args parsing
    steps = parse_steps(args)
    config = make_config(parse_config(args))

    # Calling pipeline
    pipeline(make_steps(steps, config))
