#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table.

    Args:
        env: FrozenLakeEnv instance with render_mode="ansi"
        Q: numpy.ndarray representing the trained Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards (float): total accumulated rewards
        rendered_outputs (list[str]): list of rendered board states
    """
    # Initialize the environment and get the initial state
    state = env.reset()[0]
    rendered_outputs = []
    total_rewards = 0

    for _ in range(max_steps):
        # Capture the current board state
        rendered_outputs.append(env.render())

        # Select the best action using the Q-table
        action = np.argmax(Q[state])

        # Execute the action and observe the outcome
        next_state, reward, done, _, _ = env.step(action)

        # Accumulate the reward
        total_rewards += reward

        # Move to the next state
        state = next_state

        # Exit the loop if the episode is finished
        if done:
            break

    # Render the final board state after the episode ends
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
