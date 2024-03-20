#!/usr/bin/env python3
"""
function that calculates a correlation matrix
"""

import numpy as np


def correlation(C):
    """
    Function that calculates a correlation matrix:
    Argument:
        - C is a numpy.ndarray of shape (d, d) containing a covariance matrix:
            * d is the number of dimensions
    Returns: a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    # Check if C is a numpy.ndarray
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check if C is a 2D square matrix
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Check if C is a symmetric matrix
    d = C.shape[0]

    # Calculate the correlation matrix
    Co = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            Co[i][j] = C[i][j] / np.sqrt(C[i][i] * C[j][j])

    # Return the correlation matrix
    return Co
