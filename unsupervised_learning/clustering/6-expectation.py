#!/usr/bin/env python3
""" Expectation step in the EM algorithm for a GMM """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions
        - pi: np.ndarray (k,) priors for each cluster
            - k: number of clusters
        - m: np.ndarray (k, d) centroid means for each cluster
        - S: np.ndarray (k, d, d) covariance matrices for each cluster
    Returns: g, l, or None, None on failure
        - g: np.ndarray (k, n) posterior probs for each data point in each cluster
        - l: float total log likelihood
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    # Extracting dimensions
    n, d = X.shape
    k = pi.shape[0]

    # Further validation
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    # Checking priors sum to 1
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    # Initializing array for posterior probabilities
    g = np.zeros((k, n))

    # Calculating posterior probabilities
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    # Calculating total log likelihood
    l = np.sum(np.log(np.sum(g, axis=0)))

    # Normalizing posterior probabilities
    g = g / np.sum(g, axis=0)

    # Return posterior probabilities and total log likelihood
    return g, l
