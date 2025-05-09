#!/usr/bin/env python3
""" trains an agent that can play an episode of FrozenLake """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode with a trained agent.

    Arguments:
        - env: FrozenLakeEnv instance (with render_mode="ansi")
        - Q: numpy.ndarray containing the Q-table
        - max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: float, sum of rewards obtained
        rendered_outputs: list of str, each board row or action line
    """
    state, _ = env.reset()
    total_rewards = 0
    actions = ["Left", "Down", "Right", "Up"]
    rendered_outputs = []
    # Capture map description once
    desc = env.desc.tolist()

    def board_lines(s):
        """
        Return list of ASCII strings for each row with agent at state s.
        """
        lines = []
        for i, row in enumerate(desc):
            line = ""
            for j, col in enumerate(row):
                cell = col.decode("utf-8")
                idx = i * len(row) + j
                if idx == s:
                    line += f'"{cell}"'
                else:
                    line += cell
            lines.append(line + '\n')
        return lines

    for _ in range(max_steps):
        # Append current board rows
        for ln in board_lines(state):
            rendered_outputs.append(ln)
        # Choose best action
        action = np.argmax(Q[state])
        # Append action line
        rendered_outputs.append(f"  ({actions[action]})\n")
        # Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state
        if terminated or truncated:
            break

    # Append final board rows
    for ln in board_lines(state):
        rendered_outputs.append(ln)

    return total_rewards, rendered_outputs
