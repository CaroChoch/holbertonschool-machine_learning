#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using the Bayesian
Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion
    Arguments:
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions for each data point
        - kmin: positive integer containing the minimum number of clusters
        to check for (inclusive)
        - kmax: positive integer containing the maximum number of clusters
        to check for (inclusive)
        - iterations: positive integer containing the maximum number of
        iterations for the EM algorithm
        - tol: a non-negative float containing the tolerance for the EM
        algorithm
        - verbose: boolean that determines if the EM algorithm should print
        information to the standard output
    Returns: best_k, best_result, l, b, or None, None, None, None on failure :
        - best_k is the best value for k based on its BIC
        - best_result is tuple containing pi, m, S
            - pi is a numpy.ndarray of shape (k,) containing the cluster priors
            for the best number of clusters
            - m is a numpy.ndarray of shape (k, d) containing the centroid
            means for the best number of clusters
            - S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for the best number of clusters
        - l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
        likelihood for each cluster size tested
        - b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested
            - Use: BIC = p * ln(n) - 2 * l
            - p is the number of parameters required for the model
            - n is the number of data points used to create the model
            - l is the log likelihood of the model
    """
    # Input Validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Get the number of data points and dimensions
    n, d = X.shape
    # Initialize the log likelihood and BIC arrays
    log_likelihood = []
    # Initialize the best k, result, and BIC
    b = []
    best_k = None
    best_result = None
    best_bic = float('inf')

    # Iterate over the range of clusters
    for k in range(kmin, kmax + 1):
        # Run the EM algorithm
        pi, m, S, _, like = expectation_maximization(
            X, k, iterations, tol, verbose)
        # Calculate the number of parameters
        p = k + k * d + k * d * (d + 1) // 2 - 1
        # Store the log likelihood
        log_likelihood.append(like)
        # Calculate the BIC
        b.append(p * np.log(n) - 2 * like)
        # Update the best BIC
        if b[-1] < best_bic:
            best_bic = b[-1]
            best_k = k
            best_result = (pi, m, S)
    # Convert the log likelihood and BIC to numpy arrays
    log_likelihood = np.array(log_likelihood)
    # Convert the BIC to a numpy array
    b = np.array(b)

    return best_k, best_result, log_likelihood, b
