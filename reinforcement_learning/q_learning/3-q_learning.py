#!/usr/bin/env python3
"""Performs Q-learning"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning
    Arguments:
        - env is the FrozenLakeEnv instance
        - Q is a numpy.ndarray containing the Q-table
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
        - epsilon is the initial threshold for epsilon greedy
        - min_epsilon is the minimum value that epsilon should decay to
        - epsilon_decay is the decay rate for updating epsilon between episodes
    Returns:
        Q, total_rewards
        - Q is the updated Q-table
        - total_rewards is a list containing the rewards per episode
    """
    total_rewards = []  # List to store total rewards for each episode
    max_epsilon = epsilon  # Initial epsilon value

    for episode in range(episodes):
        state, _ = env.reset()  # Reset the environment for a new episode
        rewards_current_episode = 0  # Initialize total reward for this episode

        for step in range(max_steps):
            # Select an action using epsilon-greedy strategy
            action = epsilon_greedy(Q, state, epsilon)
            # Execute the action in the environment
            observation, reward, terminated, _, _ = env.step(action)

            # If the agent falls in a hole, assign a negative reward
            if terminated and reward == 0:
                reward = -1

            # UPDATE THE Q-VALUE USING THE Q-LEARNING FORMULA
            # Curent Q-value
            old_value = Q[state, action]
            # Maximum Q-value for the next state
            next_max = np.max(Q[observation, :])
            # Q-learning update calculation
            update = reward + gamma * next_max
            # Update the Q-value using Bellman equation
            Q[state, action] = (1 - alpha) * old_value + alpha * update

            state = observation  # Move to the next state
            rewards_current_episode += reward  # Accumulate reward

            if terminated:  # If the episode is terminated, exit the loop
                break

        # Decay epsilon to reduce exploration over time
        decay_factor = np.exp(-epsilon_decay * episode)
        epsilon_diff = max_epsilon - min_epsilon
        epsilon = min_epsilon + epsilon_diff * decay_factor

        # Append the total reward of this episode to the list
        total_rewards.append(rewards_current_episode)

    return Q, total_rewards
