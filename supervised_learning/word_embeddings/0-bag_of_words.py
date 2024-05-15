#!/usr/bin/env python3
""" Creates a bag of words embedding matrix """
import numpy as np
import re

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
    # Tokenize sentences into words and normalize by lowercasing and removing punctuation
    tokenized_sentences = [re.findall(r'\b\w+\b', sentence.lower()) for sentence in sentences]

    # If no vocab is provided, create vocab from all unique words in the sentences
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))

    # Create a word-to-index mapping
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    # Initialize the embedding matrix with zeros
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Populate the embedding matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    # The list of features is the vocabulary used
    features = vocab

    return embeddings, features


