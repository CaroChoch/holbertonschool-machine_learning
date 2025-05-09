#!/usr/bin/env python3
""" plays an episode of FrozenLake using Q-learning """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode with a Q-learning agent.

    Arguments:
        - env: FrozenLakeEnv instance (with render_mode="ansi")
        - Q: numpy.ndarray containing the Q-table
        - max_steps: maximum number of steps to play

    Returns:
        total_rewards: float, sum of rewards obtained
        rendered_outputs: list of str, each element is either ""
                          (to create the initial blank line),
                          the full ANSI-rendered board,
                          or an action line like "  (Down)"…
    """
    # Reset the environment and get the initial state
    state = env.reset()[0]
    total_rewards = 0
    actions = ["Left", "Down", "Right", "Up"]

    # Start with a blank line to match the blank line
    rendered_outputs = [""]

    for _ in range(max_steps):
        # Capture and store the full board rendering
        rendered_outputs.append(env.render())

        # Select the best action based on the Q-table
        action = int(np.argmax(Q[state]))
        rendered_outputs.append(f"  ({actions[action]})")

        # Take the action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state

        # If episode is over, render the final board and exit loop
        if terminated or truncated:
            rendered_outputs.append(env.render())
            break

    # Close the environment and return results
    env.close()
    return total_rewards, rendered_outputs
