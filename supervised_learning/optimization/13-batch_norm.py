#!/usr/bin/env python3
"""
Function that normalizes an unactivated output of a neural
network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output of a neural
    network using batch normalization
    Arguments:
     - Z is a numpy.ndarray of shape (m, n) that should be normalized
        * m is the number of data points
        * n is the number of features in Z
     - gamma is a numpy.ndarray of shape (1, n) containing the scales
        used for batch normalization
     - beta is a numpy.ndarray of shape (1, n) containing the offsets
        used for batch normalization
     - epsilon is a small number used to avoid division by zero
    Returns:
     The normalized Z matrix
    """
    # Calculate the mean along each feature (column) of Z
    mean = np.mean(Z, axis=0)
    # Calculate the variance along each feature (column) of Z
    variance = np.var(Z, axis=0)
    # Normalize Z using batch normalization formula
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    # Scale and shift the normalized Z using gamma and beta respectively
    normalized_output = gamma * Z_norm + beta

    return normalized_output
