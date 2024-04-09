#!/usr/bin/env python3
""" Absorbing Chains """
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    Argument:
        - P: is a is a square 2D numpy.ndarray of shape (n, n)
        representing the standard transition matrix
            - P[i, j] is the probability of transitioning from
            state i to state j
            - n is the number of states in the markov chain
    Return: True if it is absorbing, or False on failure
    """
    # Input Validation : Return None if P is not a 2D numpy array
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    # Input validation : Return None if P is not a square matrix
    if P.shape[0] != P.shape[1]:
        return None
    # Number of states
    n = P.shape[0]

    # Check if any row in the matrix equals the corresponding row
    # of the identity matrix
    for i in range(n):
        # The np.eye() function from the NumPy library creates an
        # identity matrix. An identity matrix is a square matrix in
        # which all elements on the main diagonal are equal to 1, and
        # all other elements are equal to 0.
        if np.all(P[i] == np.eye(n)[i]):
            return True  # If an absorbing state is found, return True
    return False  # If no absorbing state is found, return False
