#!/usr/bin/env python3
"""
mean and covariance of a data set
"""

import numpy as np


def mean_cov(X):
    """
    calculates the mean and covariance of a data set
    Argument:
        - X: numpy.ndarray of shape (n, d) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
    Returns: mean, cov
    """
    # Check if X is a numpy.ndarray of shape (n, d)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Check if X contains multiple data points
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    # Get the number of data points (n) and dimensions (d)
    n, d = X.shape

    # Calculate the mean and covariance of the data set
    mean = np.mean(X, axis=0).reshape(1, d)
    cov = np.dot((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
