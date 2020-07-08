import argparse
import logging

from billys.pipeline import pipeline, make_steps, get_config, get_default_steps
from billys.util import get_log_level


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    # parser.add_argument('--use-checkpoints',
    #                     metavar='use_checkpoints',
    #                     type=bool,
    #                     nargs='?',
    #                     default=True,
    #                     help='''enable or disable checkpoints. Note that if you disable checkpoints,
    #                             your pipeline should start from step 0.''')

    parser.add_argument('--steps', metavar='steps', type=str, default=get_default_steps(),
                        nargs='*', help='Pipeline steps')

    parser.add_argument('--wdir', metavar='data_home', type=str, default=None,
                        nargs='?', help='Working directory')

    parser.add_argument('--dataset', metavar='checkpoint_name', type=str, default=None,
                        nargs='?', help='Checkpoint name')

    parser.add_argument('--dump', metavar='dump_name', type=str, default=None,
                        nargs='?', help='Dump name')

    parser.add_argument('--good', metavar='force_good', type=str, default=None,
                        nargs='?', help='Force dataset as good')

    parser.add_argument('--homography', metavar='homography_model_path', type=str, default=None,
                        nargs='?', help='Homography model path')

    parser.add_argument('--log-level', metavar='log_level', type=str,
                        choices=['critical', 'error',
                                 'warning', 'info', 'debug'],
                        default='info',
                        nargs='?', help='Working directory')

    args = parser.parse_args()

    logging.basicConfig(level=get_log_level(args.log_level))
    logging.debug(args)

    steps = args.steps
    config = get_config(data_home=args.wdir,
                        checkpoint_name=args.dataset,
                        dump_name=args.dump,
                        force_good=args.good,
                        homography_model_path=args.homography)

    pipeline(make_steps(steps, config))
