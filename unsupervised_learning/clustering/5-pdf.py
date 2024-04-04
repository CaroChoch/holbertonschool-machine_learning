#!/usr/bin/env python3
""" Probability Density Function (PDF) of a gaussian distribution """
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
        Arguments:
        - X: np.ndarray (n, d) of data points
            - n: number of data points
            - d: number of dimensions
        - m: np.ndarray (d,) of the mean of the distribution
        - S: np.ndarray (d, d) of the covariance matrix of the distribution
        Returns: P, np.ndarray (n,) of the PDF values for each data point
    """
    # Checking if X is a numpy array with two dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    # Checking if m is a numpy array with one dimension
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    # Checking if S is a numpy array with two dimensions
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    # Extracting the number of data points (n) and dimensions (d) from X
    n, d = X.shape

    # Checking if the dimensions of m and S match the dimensions of X
    if d != m.shape[0] or d != S.shape[0] or d != S.shape[1]:
        return None

    # Calculate determinant and inverse of S
    det = np.linalg.det(S)
    if det <= 0:
        return None
    inv = np.linalg.inv(S)

    # Calculate exponent term
    exponent = -0.5 * np.sum((X - m) @ inv * (X - m), axis=1)

    # Calculate normalization factor
    norm_factor = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))

    # Calculate PDF values
    P = norm_factor * np.exp(exponent)

    # Set minimum value to the PDF
    P = np.maximum(P, 1e-300)

    return P
