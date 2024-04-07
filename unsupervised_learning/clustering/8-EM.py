#!/usr/bin/env python3
""" expectation maximization for a GMM """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    Arguments:
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions
        - k: positive int, number of clusters
        - iterations: positive int, number of iterations
        - tol: non-negative float, tolerance of log likelihood
        - verbose: bool for printing information
            - i: is the nu;ber of iterations of the EM algorithm
            - l: is the log likelihood, rounded to 5 decimal places
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
          pi: np.ndarray (k,) of priors for each cluster
          m: np.ndarray (k, d) of centroid means for each cluster
          S: np.ndarray (k, d, d) of covariance matrices for each cluster
          g: np.ndarray (k, n) of probabilities for each data point in each
          cluster
          l: log likelihood of the model
    """
    # Input Validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialization
    pi, m, S = initialize(X, k)
    g, l = expectation(X, pi, m, S)

    # Store the previous log likelihood
    prev_l = l

    # EM iterations
    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, l = expectation(X, pi, m, S)

        # Verbose mode: printing log likelihood after every 10 iterations
        if verbose:
            if i % 10 == 0:
                print('Log Likelihood after {} iterations: {}'.format(
                    i, l), np.round(l, 5))
        # Check convergence
        if abs(l - prev_l) <= tol:
            break
        prev_l = l

    # Final log likelihood
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i+1, l), np.round(l, 5))
    return pi, m, S, g, l
