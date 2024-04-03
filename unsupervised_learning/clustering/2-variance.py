#!/usr/bin/env python3
""" Variance """
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    Arguments:
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    Returns: var, or None on failure
        var is the total variance
    """
    # Checking if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    # Checking if C is a numpy array with two dimensions
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    # Checking if number of dimensions of X matches number of dimensions of C
    if X.shape[1] != C.shape[1]:
        return None

    # Calculating distances between each data point and centroids
    distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    # Finding the minimum distance per data point
    min_dist = np.min(distances, axis=0)
    # Calculating the total variance
    var = np.sum(min_dist ** 2)
    return var
