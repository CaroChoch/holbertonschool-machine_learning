#!/usr/bin/env python3
""" PCA v2"""
import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset
    Arguments:
        - X is a numpy.ndarray of shape (n, d) where:
            * n is the number of data points
            * d is the number of dimensions in each point
            * all dimensions have a mean of 0 across all data points
        - ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
        version of X
    """
    # Perform Singular Value Decomposition (SVD) of the input matrix X.
    # The result consists of three components:
    #   - u: Left singular vectors matrix
    #   - s: Singular values array
    #   - vh: Right singular vectors matrix
    # Mean centering
    X_centered = X - np.mean(X, axis=0)

    # Singular Value Decomposition (SVD)
    u, s, vh = np.linalg.svd(X_centered)
    
    # Construct transformation matrix T
    T = - np.dot(X_centered, vh[:ndim].T)
    
    return T
    