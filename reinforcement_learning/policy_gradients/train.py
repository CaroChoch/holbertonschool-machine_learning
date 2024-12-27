#!/usr/bin/env python3
""" implements the training of a neural network model using policy gradients """
import numpy as np
import gym
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Function that implements a full training
    Arguments:
        - env is the initial environment
        - nb_episodes is the number of episodes used for training
        - alpha is the learning rate
        - gamma is the discount factor
        - show_result is a boolean to determine if the environment will be rendered every 1000 episodes
    Returns:
        - all values of the score (sum of all rewards during one episode loop)
    """
    # Initialize the weights
    weights = np.random.rand(*env.observation_space.shape, env.action_space.n)

    # Initialize the list to store scores
    total_scores = []

    for episode_num in range(1, nb_episodes + 1):
        current_state = env.reset()[None, :]
        cumulative_gradient = 0
        episode_score = 0
        episode_done = False

        while not episode_done:
            # Compute the action and the gradient
            action, action_gradient = policy_gradient(current_state, weights)

            # Take a step in the environment
            next_state, reward, episode_done, info = env.step(action)
            next_state = next_state[None, :]

            # Update the episode score
            episode_score += reward

            # Accumulate the gradient
            cumulative_gradient += action_gradient

            # Update the weights
            weights += (alpha * cumulative_gradient
                        * (reward + gamma * np.max(next_state.dot(weights))
                           * (not episode_done) - current_state.dot(weights)[0, action]))

            # Update the state
            current_state = next_state

        # Store the episode score
        total_scores.append(episode_score)

        # Print the current episode number and the score
        print("Episode: {}, Score: {}".format(
            episode_num, episode_score), end="\r", flush=False)

        # Render the environment every 1000 episodes
        if show_result and episode_num % 1000 == 0:
            env.render()

    return total_scores
