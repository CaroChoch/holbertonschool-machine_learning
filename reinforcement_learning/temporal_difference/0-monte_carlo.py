#!/usr/bin/env python3
""" Monte Carlo Algorithm """
import numpy as np
import gym


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo Algorithm
    Arguments:
        - env: the openAI environment instance
        - V: np.ndarray of shape(s,) containing the value estimate
        - policy: function that takes in a state and returns the next action
        - episodes: the total number of episodes to train over
        - max_steps: the maximum number of steps per episode
        - alpha: the learning rate
        - gamma: the discount rate
    Returns: V, the updated value estimate
    """

    for ep in range(episodes):
        state = env.reset()  # reset environment to starting state
        # initialize list to store states and rewards in this episode
        episode = []
        done = False  # boolean to indicate if episode is done
        G = 0  # initialize return G

        for step in range(max_steps):
            action = policy(state)  # choose action based on policy

            # take action in the environment
            new_state, reward, done, _ = env.step(action)
            episode.append([state, reward])  # store state and reward
            if done:
                break
            state = new_state  # update state

        episode = np.array(episode, dtype=int)

        for s in reversed(range(len(episode))):
            state, reward = episode[s]
            # calculate return G with discount factor gamma
            G = gamma * G + reward

            # check if state is first visited in this episode
            if state not in episode[:ep, 0]:
                V[state] += alpha * (G - V[state])  # update value function

    # Return the updated value estimate
    return V
