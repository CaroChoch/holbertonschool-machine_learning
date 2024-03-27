#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset
    Arguments:
        - X is a numpy.ndarray of shape (n, d) where:
            * n is the number of data points
            * d is the number of dimensions in each point
            * all dimensions have a mean of 0 across all data points
        - var is the fraction of the variance that the PCA transformation
            should maintain
    Returns: the weights matrix, W, that maintains var fraction of X‘s
        original variance
    """
    # Perform Singular Value Decomposition (SVD) of the input matrix X.
    # The result consists of three components:
    #   - u: Left singular vectors matrix
    #   - s: Singular values array
    #   - vh: Right singular vectors matrix
    u, s, vh = np.linalg.svd(X)

    # Calculate cumulative explained variance
    cum_var = np.cumsum(s)

    # Calculate threshold for variance
    threshold = cum_var[-1] * var

    # Determine the number of components to retain
    r = np.argwhere(cum_var >= threshold)[0, 0]

    # Construct the weight matrix W, containing the principal components
    W = vh[:r + 1].T

    return W
