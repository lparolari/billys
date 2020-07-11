import logging

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
