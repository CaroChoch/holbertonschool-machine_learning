#!/usr/bin/env python3
""" Creates a TF-IDF embedding """
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    # Checking if vocab is provided, if not, using all words from sentences
    if vocab is None:
        # Initializing TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        # Fitting the vectorizer and transforming sentences into vectors
        X = vectorizer.fit_transform(sentences)
        # Getting the vocabulary from the vectorizer
        vocab = vectorizer.get_feature_names()
    else:
        # Initializing TF-IDF vectorizer with provided vocab
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        # Fitting the vectorizer and transforming sentences into vectors
        X = vectorizer.fit_transform(sentences)

    # Converting the sparse matrix X into a dense numpy array
    embeddings = X.toarray()
    # Assigning the vocabulary to the features list
    features = vocab

    # Returning the embeddings and features
    return embeddings, features
