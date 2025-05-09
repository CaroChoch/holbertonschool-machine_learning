#!/usr/bin/env python3
""" Implements the training of a policy gradient model """
import numpy as np


# Import the policy_gradient function using __import__ as required
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045,
          gamma=0.98, show_result=False):
    """
    Trains a model using policy gradients.

    Arguments:
    - env: initialized Gymnasium environment
    - nb_episodes: number of episodes for training
    - alpha: learning rate
    - gamma: discount factor
    - show_result: if True, render env every 1000 episodes

    Returns:
    - List of scores (sum of rewards) per episode
    """
    # Initialize weights randomly: shape (state_dim, action_dim)
    weights = np.random.rand(*env.observation_space.shape,
                             env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        # Reset environment (returns obs, info)
        state, info = env.reset()
        cumulative_gradient = 0
        episode_score = 0
        done = False

        while not done:
            # Compute action and its gradient
            action, grad = policy_gradient(state, weights)
            # Take the action
            # Gymnasium returns (obs, reward, terminated,
            #                       truncated, info)
            next_obs, reward, terminated, truncated, info = \
                env.step(action)
            done = terminated or truncated
            episode_score += reward
            cumulative_gradient += grad

            # Compute TD target and error
            td_target = (
                reward
                + gamma * np.max(next_obs.dot(weights)) * (not done)
            )
            td_error = td_target - state.dot(weights)[action]

            # Update weights along the policy gradient
            weights += alpha * cumulative_gradient * td_error
            state = next_obs

        scores.append(episode_score)
        # Print episode number and score
        print(f"Episode: {episode} Score: {float(episode_score)}")

        # Render environment every 1000 episodes if requested
        if show_result and (episode + 1) % 1000 == 0:
            env.render()

    return scores
