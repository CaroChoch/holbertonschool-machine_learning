#!/usr/bin/env python3
"""Play an episode in the FrozenLake environment using the trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Args:
        env (gymnasium.Env): FrozenLakeEnv instance with render_mode="ansi"
        Q (numpy.ndarray): trained Q-table
        max_steps (int): maximum number of steps for the episode

    Returns:
        total_rewards (float): sum of rewards obtained during the episode
        rendered_outputs (list[str]): list of ANSI-rendered board states and actions at each step
    """
    # Reset the environment and get the initial state
    state, _ = env.reset()
    rendered_outputs = []
    total_rewards = 0.0

    # Map action indices to human-readable names
    action_mapping = {
        0: "Left",
        1: "Down",
        2: "Right",
        3: "Up"
    }

    for _ in range(max_steps):
        # Render the current board as ANSI text and record it
        board = env.render()
        rendered_outputs.append(board)
        # Select the greedy action from the Q-table
        action = int(np.argmax(Q[state]))
        # Record the chosen action in parentheses
        rendered_outputs.append(f"  ({action_mapping[action]})")

        # Perform the action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state

        # If the episode is over (goal reached or hole), render final state and exit
        if terminated or truncated:
            board = env.render()
            rendered_outputs.append(board)
            break
    else:
        # If max_steps reached without termination, render the final state
        board = env.render()
        rendered_outputs.append(board)

    return total_rewards, rendered_outputs
