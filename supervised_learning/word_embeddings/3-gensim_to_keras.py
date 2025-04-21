#!/usr/bin/env python3
""" Convert a gensim word2vec model to a keras Embedding layer """
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    Arguments:
        - model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
