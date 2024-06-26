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
    # Extract the number of states and actions from the environment
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the Q-table with zeros
    q_table = np.zeros((num_states, num_actions))

    return q_table
