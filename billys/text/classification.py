# from sklearn.feature_extraction.text import CountVectorizer

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(dataset.data)
# print(X_train_counts.shape)


# from sklearn.feature_extraction.text import TfidfTransformer
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# X_train_tf.shape

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tfidf.shape

# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_tfidf, dataset.target)

# docs_new = ['gas', 'supergas premium']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, dataset.target_names[category]))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def train(data, targets, target_names):

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(data, targets.astype('int'))

    docs_new = ['gas', 'acqua']
    # X_new_counts = count_vect.transform(docs_new)
    # X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = text_clf.predict(docs_new)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, category))

    return text_clf
