#!/usr/bin/env python3
""" K-means """
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    Arguments:
        - X: numpy.ndarray of shape (n, d) containing the dataset
        - k: positive integer containing the number of clusters
        - iterations: positive integer containing the maximum number of
        iterations
    Returns: C, clss, or None, None on failure
        - C: numpy.ndarray (k, d) with the centroid means for each cluster
        - clss: numpy.ndarray (n,) with the index of the cluster in C that
              each data point belongs to
    """
    # Checking if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    # Checking if k is a positive integer
    if type(k) is not int or k <= 0:
        return None, None
    # Checking if iterations is a positive integer
    if type(iterations) is not int or iterations <= 0:
        return None, None

    # Extracting the number of data points (n) and dimensions (d) from X
    n, d = X.shape
    # Calculating the minimum and maximum values along each dimension of X
    low = X.min(axis=0)
    high = X.max(axis=0)
    # Initializing centroids with random values within the range
    C = np.random.uniform(low, high, (k, d))
    # Checking if centroids initialization failed
    if C is None:
        return None, None

    # Iterating through the specified number of iterations
    for i in range(iterations):
        # Creating a copy of the centroids to enable comparison
        C_cp = np.copy(C)
        # Assigning each data point to the nearest centroid
        clss = np.linalg.norm(X - C[:, np.newaxis], axis=2).argmin(axis=0)
        # Updating centroids based on the mean of data points in each cluster
        for j in range(k):
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(X.min(axis=0), X.max(axis=0))
            else:
                C[j] = X[clss == j].mean(axis=0)
        # Verifying convergence by checking if the centroids remain the same
        if (C == C_cp).all():
            return C, clss
