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
    # Creates a CountVectorizer instance, specifying the vocabulary if provided
    vectorizer = CountVectorizer(vocabulary=vocab)
    # Transforms the input sentences into a bag of words representation
    X = vectorizer.fit_transform(sentences)
    # Converts the bag of words representation into a dense matrix
    embeddings = X.toarray()
    # Getting the list of features (words) used in bag of words representation
    features = vectorizer.get_feature_names_out()
    # Returns the embeddings (bag of words matrix) and  the list of features
    return embeddings, features
