#!/usr/bin/env python3
""" Creates a TF-IDF embedding """
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    Arguments:
        - sentences is a list of sentences to analyze
        - vocab is a list of the vocabulary words to use for the analysis
            * If None, all words within sentences should be used
    Returns: embeddings, features
        - embeddings is a numpy.ndarray shape (s, f) containing the embeddings
            * s is the number of sentences in sentences
            * f is the number of features analyzed
        - features is a list of the features used for embeddings
    """
    # Creates a TfidfVectorizer instance, specifying the vocabulary if provided
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    # Transforms the input sentences into a TF-IDF representation
    X = vectorizer.fit_transform(sentences)
    # Converts the TF-IDF representation into a dense matrix
    embeddings = X.toarray()
    # Getting the list of features (words) used in TF-IDF representation
    features = vectorizer.get_feature_names_out()
    # Returns the embeddings (TF-IDF matrix) and  the list of features
    return embeddings, features
