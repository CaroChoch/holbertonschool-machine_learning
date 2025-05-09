#!/usr/bin/env python3
""" trains an agent that can play an episode of FrozenLake """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode.

    Arguments:
        - env: the FrozenLakeEnv instance (with render_mode="ansi")
        - Q: numpy.ndarray containing the Q-table
        - max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: float, sum of rewards obtained
        rendered_outputs: list of str, each board state and actions
    """
    state, _ = env.reset()
    total_rewards = 0
    actions = ["Left", "Down", "Right", "Up"]
    rendered_outputs = []

    def render_board(s):
        """
        Return ASCII board with agent at state s.
        """
        desc = env.desc.tolist()
        lines = []
        for i, row in enumerate(desc):
            line = ''
            for j, col in enumerate(row):
                cell = col.decode('utf-8')
                idx = i * len(row) + j
                line += f'"{cell}"' if idx == s else cell
            lines.append(line)
        return '\n'.join(lines)

    for _ in range(max_steps):
        rendered_outputs.append(render_board(state))
        action = np.argmax(Q[state])
        rendered_outputs.append(f"  ({actions[action]})")
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        if terminated or truncated:
            break

    rendered_outputs.append(render_board(state))
    return total_rewards, rendered_outputs
