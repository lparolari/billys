import re
import string
from typing import List

import nltk
import spacy
import unidecode

import it_core_news_sm


def preprocess(text: str, nlp, use_lemmatize: bool = False) -> str:
    text = to_lower(text)
    text = remove_accented_chars(text)
    text = remove_punctuation(text)
    if use_lemmatize:
        text = lemmatize(text, nlp)
    text = remove_nums(text)
    text = remove_stopwords(text)
    text = remove_shortwords(text)
    return text


def make_nlp():
    return it_core_news_sm.load()  # spacy.load('it_core_news_sm')


def download_stopwords():
    nltk.download('stopwords')


def to_lower(text: str) -> str:
    return text.lower()


def remove_accented_chars(text):
    return unidecode.unidecode(text)


def remove_punctuation(text: str) -> str:
    table = str.maketrans(string.punctuation, "".join(
        [' ' for _ in string.punctuation]))
    return text.translate(table)


def remove_nums(text: str) -> str:
    return re.sub(r'\d+', '<num>', text)


def remove_shortwords(text: str) -> str:
    return " ".join([word for word in text.split() if len(word) > 2])


def remove_stopwords(text: str) -> str:
    stopwords = set(nltk.corpus.stopwords.words('italian'))
    return " ".join([word for word in text.split() if word not in stopwords])


def lemmatize(text: str, nlp=it_core_news_sm.load()) -> str:
    """Convert words to their base form."""
    doc = nlp(text)
    return " ".join([word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc])


def tokenize(text: str) -> List[str]:
    return text.split()
