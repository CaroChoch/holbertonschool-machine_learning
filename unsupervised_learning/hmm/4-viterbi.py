#!/usr/bin/env python3
""" The Viterbi algorithm """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    markov model
    Arguments:
        - Observation: np arr (T,) with index of observation
        - Emission: np arr (N, M) of emmission probabilities
        - Transition: np arr (N, N) of transition probabilities
        - Initial: np arr (N, 1) of initial dist
    Returns: path, P or None, None on failure
    """
    # Input Validation
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    # Number of observations
    T = Observation.shape[0]
    # Input Validation
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    # N is the number of hidden states, M is the number of possible
    N, M = Emission.shape
    # Input Validation
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    # Input Validation
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Transition.shape[0] != N:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # Initialize the Viterbi path and the previous path
    viterbi_path = np.zeros((N, T))
    # The previous path is a matrix of the same dimensions as the Viterbi path
    prev = np.zeros((N, T))
    # Initialize the Viterbi path probabilities for the first observation
    viterbi_path[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Forward pass of the Viterbi algorithm
    for t in range(1, T):
        # For each state s at time t, calculate the maximum probability of
        # reaching state s at time t based on the previous time step
        for s in range(N):
            # Calculate the maximum probability of reaching state s at time t
            # from any of the previous states
            viterbi_path[s, t] = np.max(
                viterbi_path[:, t - 1] * Transition[:, s]) * \
                    Emission[s, Observation[t]]
            # Store the previous state that maximizes the probability of
            # reaching state s at time t
            prev[s, t] = np.argmax(viterbi_path[:, t - 1] * Transition[:, s])

    # Backward pass to reconstruct the most likely sequence of hidden states
    # Find the state with the maximum probability at the last observation
    P = np.max(viterbi_path[:, T - 1])
    # Initialize the most likely path with the state that maximizes the
    # probability at the last observation
    S = np.argmax(viterbi_path[:, T - 1])
    path = [S]

    # Reconstruct the most likely path by backtracking from the last obs
    for t in range(T - 1, 0, -1):
        S = int(prev[S, t])
        path = [S] + path
    return path, P
