#!/usr/bin/env python3
""" tje backward algorithm """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    Arguments:
        - Observations: np.ndarray (T,) - contains the index of the observation
            - T: number of observations
        - Emission: np.ndarray (N, M) - contains the emission probability
            - N: number of hidden states
            - M: number of all possible observations
        - Transition: np.ndarray (N, N) - contains the transition probabilities
        - Initial: np.ndarray (N, 1) - contains the initial dist
    Returns: P, B or None, None on failure
        - P: likelihood of the observations given the model
        - B: np.ndarray (N, T) - contains the backward path probabilities
            - B[i, j] is the probability of generating the future observations
            from hidden state i at time j
    """

    # Input Validation
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    # Get the number of observations
    T = Observation.shape[0]
    # Input Validation
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    # Get the number of hidden states and the number of possible observations
    N, M = Emission.shape
    # Input Validation
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    # Input Validation
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    # Input Validation
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # Initialize the backward path probabilities matrix
    B = np.zeros((N, T))
    # Initialize the last column of the backward path probabilities matrix
    B[:, T - 1] = 1

    # Backward Algorithm
    for t in range(T - 2, -1, -1):
        # For each state i at time t, calculate the backward path probability
        for i in range(N):
            # Calculate the backward path probability for state i at time t
            B[i, t] = np.sum(
                Transition[i, :] *
                Emission[:, Observation[t + 1]] *
                B[:, t + 1])

    # Calculate the likelihood of the observations
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
