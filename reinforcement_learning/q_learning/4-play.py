#!/usr/bin/env python3
""" trains an agent that can play an episode of FrozenLake """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode with a trained agent.

    Arguments:
        env: FrozenLakeEnv instance (with render_mode="ansi")
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: float, sum of rewards obtained
        rendered_outputs: list of str, the board and actions output
    """
    state = env.reset()[0]
    total_rewards = 0
    actions = ["Left", "Down", "Right", "Up"]
    rendered_outputs = []

    for _ in range(max_steps):
        # Render and store current board state
        board_str = env.render()
        rendered_outputs.append(board_str)

        # Always exploit the Q-table
        action = np.argmax(Q[state])
        rendered_outputs.append(f"  ({actions[action]})")

        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        state = new_state

        if terminated or truncated:
            # Render final state and store
            board_str = env.render()
            rendered_outputs.append(board_str)
            break

    return total_rewards, rendered_outputs
