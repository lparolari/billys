import argparse
import logging

from billys.pipeline import pipeline, make_steps, get_config, get_default_steps

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

    parser.add_argument('--steps', metavar='steps', type=str, default=get_default_steps(),
                        nargs='*', help='Pipeline steps')

    parser.add_argument('--data_home', metavar='data_home', type=str,
                        nargs='?', help='Working directory')

    parser.add_argument('--dataset_name', metavar='dataset_name', type=str,
                        nargs='?', help='Dataset name')

    parser.add_argument('--force_good', metavar='force_good', type=str,
                        nargs='?', help='Force dataset as good')

    parser.add_argument('--homography_model_path', metavar='homography_model_path', type=str,
                        nargs='?', help='Homography model path')

    args = parser.parse_args()

    steps = args.steps
    config = get_config(args.data_home,
                        args.dataset_name, args.force_good, args.homography_model_path)

    pipeline(make_steps(steps, config))
