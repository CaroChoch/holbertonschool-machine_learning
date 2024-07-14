#!/usr/bin/env python3
""" Temporal Difference Lambda Algorithm """

import numpy as np
import gym


def td_lambtha(env, V, policy, lambtha=1, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Temporal Difference Lambda Algorithm
    Arguments:
        - env: the openAI environment instance
        - V: np.ndarray of shape(s,) containing the value estimate
        - policy: function that takes in a state and returns the next action
        - lambtha: the eligibility trace factor
        - episodes: the total number of episodes to train over
        - max_steps: the maximum number of steps per episode
        - alpha: the learning rate
        - gamma: the discount rate
    Returns: V, the updated value estimate
    """

    for ep in range(episodes):
        state = env.reset()  # Reset environment to initial state
        eligibility_trace = np.zeros(V.shape)  # Initialize eligibility trace

        for step in range(max_steps):
            action = policy(state)  # Select action based on policy

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)

            # TD error
            td_error = reward + gamma * V[next_state] - V[state]

            # Decay eligibility trace
            eligibility_trace *= gamma * lambtha
            # Increment eligibility trace for current state
            eligibility_trace[state] += 1

            # Update value estimate
            V += alpha * td_error * eligibility_trace

            # break if done to initiate the return
            if done:
                break

            # otherwise, update state to next_state and continue
            state = next_state

    return V
