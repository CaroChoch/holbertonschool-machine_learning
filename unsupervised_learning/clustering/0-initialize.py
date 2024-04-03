#!/usr/bin/env python3
""" Initialize K-means """
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    Arguments:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
            used for K-means clustering
            n: number of data points
            d: number of dimensions for each data point
        k: positive integer containing the number of clusters
    Returns: numpy.ndarray of shape (k, d) containing the initialized
             centroids for each cluster, or None on failure
    """
    # Checking if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    # Checking if k is a positive integer
    if type(k) is not int or k <= 0:
        return None

    # Extracting the number of data points (n) and dimensions (d) from X
    n, d = X.shape
    # Calculating the minimum and maximum values along each dimension of X
    low = X.min(axis=0)
    high = X.max(axis=0)
    # Initializing centroids with random values within the range
    return np.random.uniform(low, high, (k, d))
