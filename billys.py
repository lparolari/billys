import argparse
import ast
import logging
import os
from typing import Optional

from billys.pipeline import get_classifier, train_naive_bayes, classify_from_dataset, classify_from_filenames, classify_from_dump
from billys.steps import revert, dump
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument(
        'stage',
        type=str, default=None, nargs='?',
        choices=['train-naive-bayes-dataset', 'classify-from-dataset',
                 'classify-from-filenames', 'classify-from-dump'],
        help='Do the training from dataset')

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
        if stage == 'train-naive-bayes-dataset':
            """
            Train naive bayes classifier on given dataset and saves the dataset
            to file as a side effect (preprocessed_dataset.pkl), then save also
            the classifier to file (trained_classifier.pkl).
            """
            clf = train_naive_bayes(**config)
            dump(clf, name='trained_classifier.pkl')

        elif stage == 'classify-from-dataset':
            """
            Perform classification with test set and show metrics.
            Manually modify `dataset_name` and `classifier` parameters
            for different results.
            """
            classify_from_dataset(
                dataset_name='billys',
                classifier=get_classifier(
                    use_deterministic=False,
                    classifier_dump_name='trained_classifier.pkl',
                    data_home=None),
                target_names=[*revert(name='target_names.pkl'), 'unknown'],
                steps=['convert_to_images',
                       'ocr', 'show_boxed_text', 'extract_text', 'preprocess_text']
            )

        elif stage == 'classify-from-filenames':
            """
            Perform classification without showing metrics.
            Manually modify `dataset_name` and `classifier` parameters
            for different results.
            """
            classify_from_filenames(
                filenames='billys',
                classifier=get_classifier(use_deterministic=True),
                target_names=['acqua', 'garbage', 'gas',
                              'luce', 'telefono', 'unknown'],
            )

        elif stage == 'classify-from-dump':
            """
            Perform classification with test set and show metrics 
            from a dataset dump.
            Manually modify `dump_name` and `classifier` parameters
            for different results.
            """
            classify_from_dump(
                dump_name='dump.pkl',
                classifier=get_classifier(use_deterministic=True),
                target_names=['acqua', 'garbage', 'gas',
                              'luce', 'telefono', 'unknown'],
            )

        # Add here custom stages (remeber to add the stage also in the cli parameter choice!)

        # elif stage == 'my-stage':
        #     ...

        else:
            raise ValueError(f'The stage {stage} is not valid.')
    else:
        logging.info(
            'Do nothing by default, run the program with `--help` to see options.')
