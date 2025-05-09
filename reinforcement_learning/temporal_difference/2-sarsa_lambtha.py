#!/usr/bin/env python3
"""Performs SARSA(\u03bb) with eligibility trace"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon-greedy policy to choose actions
    Arguments:
        - state: the current state
        - Q: np.ndarray of shape(s, a) containing the Q table
        - epsilon: the epsilon to use for the calculation
    Returns: the action to take
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(\u03bb) with eligibility traces
    Arguments:
        - env: the OpenAI environment instance
        - Q: np.ndarray of shape(s, a) containing the Q table
        - lambtha: the eligibility trace factor
        - episodes: the total number of episodes to train over
        - max_steps: the maximum number of steps per episode
        - alpha: the learning rate
        - gamma: the discount rate
        - epsilon: the initial threshold for epsilon-greedy
        - min_epsilon: the minimum value that epsilon should decay to
        - epsilon_decay: the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    epsilon_init = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility_trace = np.zeros_like(Q)

        for _ in range(max_steps):
            new_state, reward, done, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)

            # Compute TD error
            td_error = (reward + gamma * Q[new_state, new_action]
                        - Q[state, action])

            # Update eligibility trace for the current state-action pair
            eligibility_trace[state, action] += 1

            # Update Q values with eligibility traces
            Q += alpha * td_error * eligibility_trace

            # Decay all eligibility traces
            eligibility_trace *= gamma * lambtha

            if done or truncated:
                break

            state, action = new_state, new_action

        # Decay epsilon
        epsilon = (min_epsilon
                   + (epsilon_init - min_epsilon)
                   * np.exp(-epsilon_decay * ep))

    return Q
