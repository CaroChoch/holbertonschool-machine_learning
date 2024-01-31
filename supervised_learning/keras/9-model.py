#!/usr/bin/env python3
""" Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    Arguments:
        - network: model to save
        - filename: path of the file that the model should be saved to
    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loads an entire model
    Arguments:
        - filename: path of the file that the model should be loaded from
    Returns: the loaded model
    """
    loaded_model = K.models.load_model(filename)
    return loaded_model
