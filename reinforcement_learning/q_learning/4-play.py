#!/usr/bin/env python3
""" trains an agent that can play an episode of FrozenLake """
import numpy as np
import gym


def play(env, Q, max_steps=100):
    """
    plays an episode
    Arguments:
        - env: the FrozenLakeEnv instance
        - Q: numpy.ndarray containing the Q-table
        - max_steps: the maximum number of steps in the episode
    Returns: the total rewards for the episode
    """
    state, _ = env.reset()  # Reset the environment for a new episode
    total_rewards = 0  # Initialize total rewards for the episode

    # Mapping of actions to their names
    actions = ["Left", "Down", "Right", "Up"]

    def print_board(state):
        """
        Prints the current state of the board with the agent's position
        highlighted.

        Arguments:
            - state: The current state (position) of the agent in the
            environment.
        """
        # Convert the description of the environment to a list of lists
        desc = env.desc.tolist()

        # Iterate over each row in the description
        for i, row in enumerate(desc):
            # Iterate over each column in the row
            for j, col in enumerate(row):
                # Calculate the linear index of the current cell
                index = i * len(row) + j
                # If the current cell is the agent's position, highlight it
                if index == state:
                    # Print the cell content with backticks
                    print(f"`{col.decode('utf-8')}`", end='')
                else:
                    # Print the cell content normally
                    print(col.decode('utf-8'), end='')
            # Print a new line at the end of each row
            print()

    for step in range(max_steps):
        # Always exploit the Q-table
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)

        # Display the current state of the board
        print_board(state)

        # Convert the action to a readable string and print it
        print("  ({})".format(actions[action]))

        total_rewards += reward  # Accumulate the rewards

        if terminated:
            state = next_state  # Ensure the final state is printed correctly
            break

        state = next_state  # Move to the next state

    # Print the final state of the board
    print_board(state)

    return total_rewards
