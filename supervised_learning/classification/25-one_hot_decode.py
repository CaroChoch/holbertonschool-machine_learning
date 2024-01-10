#!/usr/bin/env python3
""" function that converts a one-hot matrix into a vector of labels """
import numpy as np


def one_hot_decode(one_hot):
    """
    function that converts a one-hot matrix into a vector of labels
    Arguments:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
    Returns: a numpy.ndarray with shape (m, ) containing the numeric labels
    for each example, or None on failure
    """
    # Check if one_hot is a non-empty numpy.ndarray with dimensionality
    # (classes, m)
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    # Return the numeric labels for each example
    return np.argmax(one_hot, axis=0)
