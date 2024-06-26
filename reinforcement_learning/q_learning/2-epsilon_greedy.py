#!/usr/bin/env python3
""" Epsilon-greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):

    """
    Uses epsilon-greedy to determine the next action
    Arguments:
        - Q is a numpy.ndarray containing the q-table
        - state is the current state
        - epsilon is the epsilon to use for the calculation
    Returns:
        the next action index
    """
    # Generate a random number between 0 and 1
    p = np.random.uniform(0, 1)

    # Select the action with the highest estimated value (exploitation)
    # or a random action (exploration) based on epsilon
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])

    return action
