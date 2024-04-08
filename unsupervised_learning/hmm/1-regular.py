#!/usr/bin/env python3
""" Regular Chains """
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    Arguments:
        P: np.ndarray (n, n) representing the transition matrix
            - P[i, j]: probability of transitioning from state i to state j
            - n: number of states in the markov chain
    Returns: np.ndarray (1, n) representing the steady state probabilities,
        or None on failure
    """
    # Input Validation
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    # Get the number of states in the markov chain
    n = P.shape[0]

    # Return None if any transition probability is non-positive
    if np.any(P <= 0):
        return None

    # Return None if the transition matrix is not regular
    if np.any(np.sum(P, axis=1) != 1):
        return None

    # Initialize the probability vector with uniform distribution
    prob = np.ones((1, n)) / n

    # Iterate until no improvement
    for _ in range(100):  # Assume a maximum of 100 iterations
        # Calculate the next probability vector
        prob_prev = prob
        # Multiply the current probability vector by the transition matrix
        prob = np.matmul(prob, P)

        # Check for convergence
        if np.all(prob == prob_prev):
            return prob
        # Return None if any transition probability is non-positive
        if np.any(P <= 0):
            return None

    # Return None if the transition matrix is not regular
    return None
