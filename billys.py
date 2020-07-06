import argparse

from billys import train
from billys.pipeline import Step

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--step', metavar='step', type=int, nargs='?', default=0,
                        help='the starting step for the pipeline')

    parser.add_argument('--use-checkpoints',
                        metavar='use_checkpoints',
                        type=bool,
                        nargs='?',
                        default=True,
                        help='''enable or disable checkpoints. Note that if you disable checkpoints, 
                                your pipeline should start from step 0.''')

    args = parser.parse_args()

    train.pipeline(first_step=Step(args.step),
                   use_checkpoints=args.use_checkpoints)
