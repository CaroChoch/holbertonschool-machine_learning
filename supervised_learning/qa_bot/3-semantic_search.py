#!/usr/bin/env python3
""" performs semantic search on a corpus of documents """
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents
    Arguments:
        - corpus_path: (str) the path to the corpus of reference documents on
          which to perform semantic search
        - sentence: (str) the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence
    """

    # Initialize the list of documents with the input sentence
    documents = [sentence]

    # Iterate over the files in the corpus directory
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md') is False:
            continue
        with open(corpus_path + '/' + filename) as f:
            documents.append(f.read())

    # Load the pre-trained Universal Sentence Encoder from TensorFlow Hub
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Generate embeddings for all documents
    embeddings = embed(documents)

    # Compute the similarity between the input sentence and all documents
    correlation = np.inner(embeddings, embeddings)

    # Find the index of the document most similar to the input sentence
    closest = np.argmax(correlation[0, 1:])

    # The most similar document is the one at the index closest + 1
    most_similar = documents[closest + 1]

    # Return the most similar document
    return most_similar
