#!/usr/bin/env python3
""" forward algorithm """
import numpy as np


def forward(Observations, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    Arguments:
        Observations: np arr (T,) with index of observation
        Emission: np arr (N, M) of emmission probabilities
        Transition: np arr (N, N) of transition probabilities
        Initial: np arr (N, 1) of initial dist
    Returns: P, F or None, None on failure
            P: likelihood of the obs given the model
            F: np arr (N, T) of the fwd path probs or None
    """
    # Input Validation
    if not isinstance(Observations, np.ndarray):
        return None, None
    if len(Observations.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    # Get the number of hidden states
    N, M = Emission.shape
    # Get the number of observations
    T = Observations.shape[0]

    # Check if the number of hidden states matches in the emission and
    # transition matrices
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    # Check if the number of hidden states in the initial matrix
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # Initialize forward path probabilities matrix
    F = np.zeros((N, T))

    # Forward Algorithm
    F[:, 0] = Initial.flatten() * Emission[:, Observations[0]]
    for t in range(1, T):
        F[:, t] = np.dot(F[
            :, t - 1], Transition) * Emission[:, Observations[t]]

    # Likelihood of the observations
    P = F[:, -1].sum()

    # Return the likelihood of the observations and the forward path
    # probabilities
    return P, F
