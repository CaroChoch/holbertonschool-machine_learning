#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake.
"""


import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table and records each state and action.

    Args:
        env: FrozenLakeEnv instance with render_mode="ansi"
        Q: numpy.ndarray representing the trained Q-table
        max_steps: maximum number of steps allowed in the episode

    Returns:
        total_rewards (float): total reward accumulated during the episode
        rendered_outputs (list[str]): list of ANSI-rendered states and actions
    """
    # Reset environment and get the initial state
    state = env.reset()[0]
    rendered_outputs = []
    total_rewards = 0.0

    # Render and capture the initial state
    rendered_outputs.append(env.render())

    # Map action indices to human-readable names
    action_mapping = {
        0: "Left",
        1: "Down",
        2: "Right",
        3: "Up"
    }

    for _ in range(max_steps):
        # Choose the best action (greedy)
        action = int(np.argmax(Q[state]))
        action_name = action_mapping[action]

        # Record the chosen action first
        rendered_outputs.append(f"  ({action_name})")

        # Take the action in the environment
        next_state, reward, done, truncated, _ = env.step(action)

        # Render and capture the new state after the action
        rendered_outputs.append(env.render())

        # Update total rewards and state
        total_rewards += reward
        state = next_state

        # End episode if terminated or truncated
        if done or truncated:
            break

    return total_rewards, rendered_outputs
