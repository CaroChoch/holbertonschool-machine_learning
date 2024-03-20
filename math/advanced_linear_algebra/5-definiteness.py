#!/usr/bin/env python3
"""
definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix
    Argument:
        - matrix: numpy.ndarray of shape (n, n) whose definiteness should be calculated
                  or a list of lists representing a matrix
    Returns: the string "Positive definite", "Positive semi-definite", "Negative definite",
             "Negative semi-definite", "Indefinite", or "None" if the matrix is not square
    """
    # Check if the input is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if the input matrix is square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return "None"

    # Check if the input matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return "None"

    # Check the definiteness of the matrix
    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"

    # Check the semi-definiteness of the matrix
    if np.all(np.linalg.eigvals(matrix) >= 0):
        return "Positive semi-definite"

    # Check the definiteness of the matrix
    if np.all(np.linalg.eigvals(matrix) < 0):
        return "Negative definite"

    # Check the semi-definiteness of the matrix
    if np.all(np.linalg.eigvals(matrix) <= 0):
        return "Negative semi-definite"

    # Check the indefiniteness of the matrix
    else:
        return "Indefinite"
