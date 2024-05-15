#!/usr/bin/env python3
""" Creates a bag of words embedding matrix """
import numpy as np
import re


def preprocess_sentence(sentence):
    """
    Preprocess a sentence by removing possessives and converting to lowercase.
    Arguments:
        - sentence: a string representing a sentence
    Returns: a list of words
    """
    # Remove possessive 's and convert to lowercase
    processed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
    # Extract words from the processed sentence
    return re.findall(r'\w+', processed_sentence)


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
    # Check if sentences is a list
    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    # Preprocess all sentences
    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentence = preprocess_sentence(sentence)
        preprocessed_sentences.append(preprocessed_sentence)

    # If no vocab is provided, create vocab from all unique words
    # in the sentences
    if vocab is None:
        all_words = []
        for sentence in preprocessed_sentences:
            all_words.extend(sentence)
        vocab = sorted(set(all_words))

    # Create a word-to-index mapping
    word_to_index = {}
    for idx, word in enumerate(vocab):
        word_to_index[word] = idx

    # Initialize the embedding matrix with zeros
    num_sentences = len(sentences)
    num_features = len(vocab)
    embeddings = np.zeros((num_sentences, num_features), dtype=int)

    # Populate the embedding matrix
    for i, sentence in enumerate(preprocessed_sentences):
        for word in sentence:
            if word in word_to_index:
                word_index = word_to_index[word]
                embeddings[i, word_index] += 1

    return embeddings, vocab
