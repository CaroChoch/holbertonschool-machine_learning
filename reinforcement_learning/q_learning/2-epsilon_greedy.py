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

    # Exploitation: choose among actions with highest Q-value
    if p > epsilon:
        state_actions = Q[state, :]
        max_value = np.max(state_actions)
        max_actions = np.where(state_actions == max_value)[0]
        action = np.random.choice(max_actions)
    else:
        # Exploration: choose a random action
        action = np.random.randint(0, Q.shape[1])

    return action
