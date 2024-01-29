#!/usr/bin/env python3
""" Function that converts a label vector into a one-hot matrix """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function that converts a label vector into a one-hot matrix
    Arguments:
        labels is a one-hot encoded numpy.ndarray with shape (m, classes)
        containing the labels for each instance
        classes is the number of classes
    Returns: the one-hot matrix
    """
    labels = K.utils.to_categorical(labels, classes)
    return labels
