#!/usr/bin/env python3
""" trains an agent that can play an episode of FrozenLake """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode.

    Arguments:
        - env: the FrozenLakeEnv instance
        - Q: numpy.ndarray containing the Q-table
        - max_steps: maximum number of steps in the episode

    Returns:
        total rewards for the episode and a list of rendered outputs
        for each step.
    """
    state, _ = env.reset()
    total_rewards = 0
    actions = ["Left", "Down", "Right", "Up"]
    rendered_outputs = []

    def render_board(s):
        """
        Return ASCII representation of the board with the agent at
        state s.
        """
        desc = env.desc.tolist()
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
            lines.append(line)
        return "\n".join(lines)

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)

        rendered_outputs.append(render_board(state))
        rendered_outputs.append(f"  ({actions[action]})")

        total_rewards += reward
        state = next_state
        if terminated:
            break

    # capture final state
    rendered_outputs.append(render_board(state))
    # add blank line to match expected output
    rendered_outputs.append("")
    return total_rewards, rendered_outputs
