#!/usr/bin/env python3
""" Maximization step in the EM algorithm for a GMM """
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions
        - g: np.ndarray (k, n) posterior probs for each data point in each
            cluster
    Returns: pi, m, S, or None, None, None on failure
        - pi: np.ndarray (k,) updated priors for each cluster
        - m: np.ndarray (k, d) updated centroid means for each cluster
        - S: np.ndarray (k, d, d) updated covariance matrices for each cluster
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    # Extracting dimensions
    n, d = X.shape
    k, _ = g.shape

    # Further validation
    if g.shape[1] != n:
        return None, None, None
    # Checking if posterior probabilities sum to 1
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    # Calculating updated priors
    pi = np.zeros((k,))

    # Calculating updated centroid means
    m = np.zeros((k, d))

    # Calculating updated covariance matrices
    S = np.zeros((k, d, d))

    # Updating priors
    for i in range(k):
        S[i] = np.dot(g[i] * (X - m[i]).T, (X - m[i])) / np.sum(g[i])

    # Return updated priors, centroid means, and covariance matrices
    return pi, m, S
