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
    df = read_dump(get_filename('dataset.pkl'))
    df.reset_index(inplace=True)

    print(df)

    target_names = ['acqua', 'garbage', 'gas', 'luce', 'telefonia e internet']

    # Drop invalid bills
    df.drop(df[df['filename'].str.contains(
        '2017-GR17-239588E.pdf')].index, inplace=True)
    df.drop(df[df['filename'].str.contains(
        '2017-GR17-429317E.pdf')].index, inplace=True)
    df.drop(df[df['filename'].str.contains(
        '2019-GR19-274322E.pdf')].index, inplace=True)

    # Re-classifying wrong bills

    # Le seguenti fatture sono classificate come LUCE in realtà sono TELEFONO:
    #  2019 RegFatt00036 -- fattura n 1974197327 (6860).pdf
    #  2019 RegFatt00037 -- fattura n 1988002029 (6861).pdf
    #  2019 RegFatt00038 -- fattura n 1989002224 (6862).pdf
    #  2019 RegFatt00039 -- fattura n 1989404980 (6863).pdf

    # La seguente da TELEFONO va messa in ENERGIA:
    #  2019 RegFatt00239 -- Fattura Cliente 300680879 del 04102019 (8141).pdf

    # la seguente va ELIMINATA dal dataset in quanto la prima pagina è sbagliata:
    #  2020 RegFatt00012 -- Fattura Vodafone (8644).pdf

    # 'target'=np.where(df1['stream'] == 2, 10, 20)
    # print(df[df['filename'].str.contains(
    #     '2019 RegFatt00036 - - fattura n 1974197327 (6860).pdf')]['target'])

    print(len(df.loc[df['filename'].astype(str).str.contains(
        '2019 RegFatt00036 - - fattura n 1974197327 (6860).pdf')]))
    print(len(df.loc[df['filename'].str.contains(
        '2019 RegFatt00037 -- fattura n 1988002029 (6861).pdf')]))
    print(len(df.loc[df['filename'].str.contains(
        '2019 RegFatt00038 -- fattura n 1989002224 (6862).pdf')]))
    print(len(df.loc[df['filename'].str.contains(
        '2019 RegFatt00039 -- fattura n 1989404980 (6863).pdf')]))

    print(len(df.loc[df['filename'].str.contains(
        '2019 RegFatt00239 -- Fattura Cliente 300680879 del 04102019 (8141).pdf')]))

    print(len(df.loc[df[df['filename'].str.contains(
        '2020 RegFatt00012 -- Fattura Vodafone (8644).pdf')].index]))

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
train_from_middle()
