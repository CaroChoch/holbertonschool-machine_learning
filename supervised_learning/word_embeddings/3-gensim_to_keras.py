#!/usr/bin/env python3
""" Convert a gensim word2vec model to a keras Embedding layer """
from gensim.models import Word2Vec
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    Arguments:
        - model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
