import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train(X_train, y_train, X_test, y_test, grid_search_parameters=None):
    """
    Perform the training and test phase for the classifier.
    The classifier is a Multinomial Naive Baise model from scikit-learn.
    Classifier rarameters have been tuned with grid search.

    Parameters
    ----------
    X_train
        A list of values representing training data.
    y_train
        A list of values representing train validation data.
    X_test
        A list of values representing test data.
    y_test
        A list of values representing test validation data.
    grid_search_parameters
        A dict of key values representing parameters for
        grid search. None for defaults.

    Returns
    -------
    classifier
        A trained classifier with pipeline for text transformation.
    """

    # Text classifier pipeline.
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    # Grid search parameters.
    parameters = grid_search_parameters or {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(X_train, y_train)

    logging.info(f'Model\'s best score: {gs_clf.best_score_}')
    logging.info(f'Model\'s best params: {gs_clf.best_params_}')

    return gs_clf


def classify_bow(documents: List[str], max_word_diff: int = 1, unknown_cat_id: int = 5) -> List[int]:

    predicted = []

    for document in documents:

        document = [document]
        if document == ['']:
            # Skip empty data, return an unknown category.
            predicted.append(5)
            continue

        # use the scikit vectorized for creating the bag of words
        vectorizer = CountVectorizer().fit(document)
        bag_of_words = vectorizer.transform(document)

        # create the sum of the bag of words in order to represent frequencies
        sum_words = bag_of_words.sum(axis=0)

        # get words freq
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # create words dict
        kws_per_category = [['acqua', 'idrico', 'depurazione', 'fognatura', 'trevigiano'],
                            ['spazzatura', 'immondizia', 'riciclato',
                                'riciclo', 'rifiuti', 'rifiuto', 'savno'],
                            ['gas', 'naturale'],
                            ['luce', 'energia', 'tensione', 'potenza',
                                'elettricita', 'elettrico', 'elettrica'],
                            ['telefonia', 'telefonico', 'internet', 'ricaricabile', 'ricarica',
                             'cellulare', 'navigazione', 'telecom', 'vodafone', 'tim']]

        # runnng classification algorithm

        best_cat = unknown_cat_id
        max_freq = -1
        max_word = "XXX"

        for j in range(5):

            kws = kws_per_category[j]

            for i in range(len(kws)):
                kw = kws[i]
                for (w, f) in words_freq:
                    if (w in kw and np.abs(len(w) - len(kw)) <= max_word_diff and f > max_freq):
                        best_cat = j
                        max_freq = f
                        max_word = w

        predicted.append(best_cat)

    return predicted
