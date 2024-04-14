#!/usr/bin/env python3
""" The Baum-Welch algorithm for a hidden Markov model """

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
            # Calculate the backward path
            B[i, t] = np.sum(
                Transition[i, :] *
                Emission[:, Observation[t + 1]] *
                B[:, t + 1])

    # Calculate the likelihood of the observations
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    Arguments:
        - Observations: np.ndarray (T,) - contains the index of the observation
            - T: number of observations
        - Transition: np.ndarray (N, N) - contains the transition probabilities
            - M: number of hidden states
        - Emission: np.ndarray (N, M) - contains the emission probabilities
            - N: is the number of output states
        - Initial: np.ndarray (N, 1) - contains the initial dist
        - iterations: int - number of times the EM algorithm should iterate
    Returns: np.ndarray, np.ndarray, or None, None on failure
        - Transition: np.ndarray (N, N) - the updated transition probabilities
        - Emission: np.ndarray (N, M) - the updated emission probabilities
    """

    # Input Validation
    if (not isinstance(Observations, np.ndarray) or
            len(Observations.shape) != 1):
        return None, None
    # Get the number of observations
    T = Observations.shape[0]
    if not isinstance(T, int) or T < 2:
        return None, None

    # Input Validation
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    # Get the number of hidden states
    M = Transition.shape[0]
    if not isinstance(M, int) or M < 1:
        return None, None
    # Input Validation
    if not (np.all(Transition >= 0) and np.all(Transition <= 1)):
        return None, None
    if not np.all(np.sum(Transition, axis=1) == 1):
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    # Get the number of possible observations
    N = Emission.shape[1]
    if not isinstance(N, int) or N < 1:
        return None, None
    if not (np.all(Emission >= 0) and np.all(Emission <= 1)):
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    # Input Validation
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    # Input Validation
    if not (np.all(Initial >= 0) and np.all(Initial <= 1)):
        return None, None
    if not np.all(np.sum(Initial) == 1):
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    # Initialize variables
    N = Emission.shape[0]
    transition_copy = Transition.copy()
    emission_copy = Emission.copy()

    # Iterate the Baum-Welch algorithm
    if iterations > 365:
        iterations = 365
    for iteration in range(iterations):
        _, alpha = forward(
            Observations,
            emission_copy,
            transition_copy,
            Initial)
        _, beta = backward(
            Observations,
            emission_copy,
            transition_copy,
            Initial)
        # Compute xi (gamma transition probabilities)
        xi = np.zeros((N, N, T - 1))
        # Iterate over time steps
        for t in range(T - 1):
            # Compute denominator for xi
            denominator = np.matmul(np.matmul(
                alpha[:, t].T,
                transition_copy) * emission_copy[:, Observations[t + 1]].T,
                beta[:, t + 1])
            # Iterate over hidden states
            for hidden_state_index in range(N):
                # Compute numerator for xi
                numerator = alpha[hidden_state_index, t] *\
                            transition_copy[hidden_state_index, :] *\
                            emission_copy[:, Observations[t + 1]].T *\
                            beta[:, t + 1].T
                # Compute xi
                xi[hidden_state_index, :, t] = numerator / denominator
        # Compute gamma (state probabilities)
        gamma = np.sum(xi, axis=1)
        # Update Transition matrix
        transition_copy = np.sum(
            xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        # Compute additional column for gamma
        gamma = np.concatenate(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))), axis=1)

        # Update Emission matrix
        num_possible_emission_states = emission_copy.shape[1]
        denominator = np.sum(gamma, axis=1)
        # Iterate over hidden states
        for state_index in range(num_possible_emission_states):
            emission_copy[:, state_index] = np.sum(
                gamma[:, Observations == state_index], axis=1)
        # Normalize Emission matrix
        emission_copy = np.divide(emission_copy, denominator.reshape((-1, 1)))

    # Return updated Transition and Emission matrices
    return transition_copy, emission_copy
