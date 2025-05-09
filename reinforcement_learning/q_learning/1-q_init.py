#!/usr/bin/env python3
""" initializes the Q-table """
import numpy as np


def q_init(env):
    """
    Initializes the Q-table
    Arguments:
        - env is the FrozenLakeEnv instance
    Returns:
        Q-table as a numpy.ndarray of zeros
    """
    # Extract the number of states and actions from the environment,
    # avec fallback sur nS / nA si besoin
    num_states = getattr(env.observation_space, 'n', getattr(env, 'nS', None))
    num_actions = getattr(env.action_space,      'n', getattr(env, 'nA', None))

    # Initialise la Q-table avec des zéros
    q_table = np.zeros((num_states, num_actions))

    return q_table
