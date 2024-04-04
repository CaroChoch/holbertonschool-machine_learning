#!/usr/bin/env python3
""" Initialize GMM """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model:
    Arguments:
        - X: (n, d) np.ndarray - data set
            - n: number of data points
            - d: number of dimensions in each data point
        - k: positive int - number of clusters
    Returns: pi, m, S, or None, None, None on failure
        - pi: (k,) np.ndarray - containing the priors for each cluster
        - m: (k, d) np.ndarray - containing the centroid means for each cluster
        - S: (k, d, d) np.ndarray - containing the covariance matrices for each
            cluster
    """
    # Checking if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    # Checking if k is a positive integer
    if not isinstance(k, int) or k < 1:
        return None, None, None

    # Extracting the number of data points (n) and dimensions (d) from X
    n, d = X.shape
    # Initializing the priors for each cluster (pi) with equal probability
    pi = np.full(shape=(k,), fill_value=1/k)
    # Initializing the centroid means for each cluster (m) using K-means
    m = kmeans(X, k)[0]
    # Initialize covariance matrices for each cluster (S) as identity matrices
    S = np.full(shape=(k, d, d), fill_value=np.identity(d))
    return pi, m, S
