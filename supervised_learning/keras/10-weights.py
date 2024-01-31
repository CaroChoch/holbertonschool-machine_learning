#!/usr/bin/env python3
""" Save and Load Weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model’s weights
    Arguments:
        - network: model whose weights should be saved
        - filename: path of the file that the weights should be saved to
        - save_format: format in which the weights should be saved
    Returns: None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    loads a model’s weights
    Arguments:
        - network: model to which the weights should be loaded
        - filename: path of the file that the weights should be loaded from
    Returns: None
    """
    network.load_weights(filepath=filename)
    return None
