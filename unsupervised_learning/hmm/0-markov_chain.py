#!/usr/bin/env python3
""" Markov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular
    state after a specified number of iterations
    Arguments:
        P: np.ndarray (n, n) representing the transition matrix
            - P[i, j]: probability of transitioning from state i to state j
            - n: number of states in the markov chain
        s: np.ndarray (1, n) representing the probability of starting in each
            state
        t: positive int, number of iterations to calculate the probability
            of being in a specific state
    Returns: np.ndarray (1, n) representing the probability of being in a
        specific state after t iterations, or None on failure
    """
    # Input Validation
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    # Calculate the probability of being in a specific state after t iterations
    for i in range(t):
        s = np.matmul(s, P)

    # Return the probability of being in a specific state after t iterations
    return s
