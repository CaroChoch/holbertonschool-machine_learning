#!/usr/bin/env python3
""" trains an agent that can play an episode of FrozenLake """
import numpy as np


def play(env, Q, max_steps=100):
    """
    plays an episode
    Arguments:
        - env: the FrozenLakeEnv instance
        - Q: numpy.ndarray containing the Q-table
        - max_steps: the maximum number of steps in the episode
    Returns: the total rewards for the episode and a list of rendered outputs
    """
    state, _ = env.reset()  # Reset the environment for a new episode
    total_rewards = 0  # Initialize total rewards for the episode
    actions = ["Left", "Down", "Right", "Up"]
    rendered_outputs = []

    def print_board(state):
        """
        Prints the current state of the board with the agent's position
        highlighted.

        Arguments:
            - state: The current state (position) of the agent in the
              environment.
        Returns: the string representation of the board
        """
        desc = env.desc.tolist()
        board = ""
        for i, row in enumerate(desc):
            for j, col in enumerate(row):
                index = i * len(row) + j
                cell = col.decode('utf-8')
                if index == state:
                    board += f"`{cell}`"
                else:
                    board += cell
            board += "\n"
        print(board, end='')
        return board

    for step in range(max_steps):
        # Always exploit the Q-table
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)

        # Display and capture the current board state
        board_str = print_board(state)
        rendered_outputs.append(board_str)

        # Display and capture the action taken
        action_str = f"  ({actions[action]})\n"
        print(action_str, end='')
        rendered_outputs.append(action_str)

        total_rewards += reward  # Accumulate the rewards
        state = next_state  # Move to the next state
        if terminated:
            break

    # Print and capture the final state of the board
    board_str = print_board(state)
    rendered_outputs.append(board_str)

    return total_rewards, rendered_outputs
