#!/usr/bin/env python3
""" Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    Arguments:
        - X is a numpy.ndarray of shape (n, d) containing the data set
        - kmin is a positive integer containing the minimum number of clusters
            to check for (inclusive)
        - kmax is a positive integer containing the maximum number of clusters
            to check for (inclusive)
        - iterations is a positive integer containing the maximum number of
            iterations for K-means
    Returns: results, d_vars, or None, None on failure
        - results is a list containing the outputs of K-means for each cluster
            size
        - d_vars is a list containing the difference in variance from the
            smallest cluster size for each cluster size
    """
    # Checking if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    # Checking if kmin is a positive integer
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    # Checking if kmax is a positive integer
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    # Checking if kmax is greater than kmin
    if kmax <= kmin:
        return None, None
    # Checking if iterations is a positive integer
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    # initializing results and d_vars lists
    results = []
    d_vars = []
    # Iterating through the range of clusters
    for k in range(kmin, kmax + 1):
        # Running K-means on the dataset
        C, clss = kmeans(X, k, iterations)
        # Checking if K-means failed
        results.append((C, clss))
        # Calculating the variance for the smallest cluster size
        if k == kmin:
            var_min = variance(X, C)
        # Calculating the difference in variance from the smallest cluster size
        d_vars.append(var_min - variance(X, C))
    # Returning the results and difference in variance
    return results, d_vars
