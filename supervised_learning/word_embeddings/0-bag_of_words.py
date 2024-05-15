#!/usr/bin/env python3
""" Creates a bag of words embedding matrix """
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
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
        # Initializing CountVectorizer
        vectorizer = CountVectorizer()
        # Fitting the vectorizer and transforming sentences into vectors
        X = vectorizer.fit_transform(sentences)
        # Getting the vocabulary from the vectorizer
        vocab = vectorizer.get_feature_names_out()
    else:
        # Initializing CountVectorizer with provided vocab
        vectorizer = CountVectorizer(vocabulary=vocab)
        # Fitting the vectorizer and transforming sentences into vectors
        X = vectorizer.fit_transform(sentences)
    # Converting the sparse matrix X into a dense numpy array
    embedding = X.toarray()
    # Assigning the vocabulary to the features list
    features = vocab

    # Returning the embeddings and features
    return embedding, features
