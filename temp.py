"""
IGNORE THIS FILE
----------------

Temporary file for intermediate dataframe preprocesssing.
"""


import random
from billys.util import read_dump, save_dump, get_filename
import numpy as np


def build_new_dataset():
    """
    Costruisce il dataset giusto dato il primo dataset con i dati dell'ocr.
    """
    df = read_dump(get_filename('image_wothout_brightness.pkl'))
    df.reset_index(inplace=True)

    print(df)

    target_names = ['acqua', 'garbage', 'gas', 'luce', 'telefonia e internet']

    subsets = []
    data_target_names = []

    for index, row in df.iterrows():

        subsets.append('train' if random.random() < 0.7 else 'test')
        data_target_names.append(target_names[row['target']])

    df['subset'] = subsets
    df['target_names'] = data_target_names

    print(df[df['subset'] == 'train'])
    print(df[df['subset'] == 'test'])

    save_dump(df, filename=get_filename('dataset_DEFINITIVO.pkl'))


def train_from_middle():
    """
    Allena il classificatore sul dataset estratto dalla funzione precedente, 
    ovvero un dump del dataframe con l'ocr.
    """
    df = revert('dataset_DEFINITIVO.pkl')

    df = extract_text(df)
    df = preprocess_text(df)

    clf = train_classifier(df)

    # dump(target_names, name=targets_dump_name)
    dump(clf, name='trained_classifier_DEFINITIVO.pkl')

    return clf


build_new_dataset()
