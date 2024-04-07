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
    g, log_likelihood = expectation(X, pi, m, S)

    # Store the previous log likelihood
    prev_l = 0

    # EM iterations
    for i in range(iterations):
        # Verbose mode: printing log likelihood after every 10 iterations
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, log_likelihood.round(5)))
        # Maximization step
        pi, m, S = maximization(X, g)
        # Expectation step
        g, log_likelihood = expectation(X, pi, m, S)

        # Check convergence
        if np.abs(log_likelihood - prev_l) <= tol:
            break
        # Update previous log likelihood
        prev_l = log_likelihood

    # Final log likelihood
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i+1, log_likelihood.round(5)))
    return pi, m, S, g, log_likelihood
