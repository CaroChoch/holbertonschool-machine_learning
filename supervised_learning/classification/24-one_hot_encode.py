#!/usr/bin/env python3
import numpy as np
""" function that converts a numeric label vector into a one-hot matrix """


def one_hot_encode(Y, classes):
    """
    function that converts a numeric label vector into a one-hot matrix
    Arguments:
        Y: input labels
        classes: number of classes
    Returns: the one-hot matrix
    """

    # Check if Y is a non-empty NumPy array
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    # Check if classes is a positive int greater than the maximum value in Y
    if type(classes) is not int or classes <= np.amax(Y):
        return None
    # Create a matrix filled with zeros of shape (classes, len(Y))
    one_hot = np.zeros((classes, Y.shape[0]))
    # Set the corresponding elements to 1 in the one-hot matrix
    one_hot[Y, np.arange(Y.shape[0])] = 1
    # Return the resulting one-hot matrix
    return one_hot
