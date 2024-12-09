#!/usr/bin/env python3
""" Performs SARSA(λ) with eligibility trace """
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
    # Generate a random number between 0 and 1
    p = np.random.uniform(0, 1)

    if p > epsilon:
        # Exploit : Choose the action with the highest Q value
        action = np.argmax(Q[state, :])
    else:
        # Explore : Choose a random action
        action = np.random.randint(0, Q.shape[1])

    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(λ) with eligibility trace
    Arguments:
        - env: the openAI environment instance
        - Q: np.ndarray of shape(s, a) containing the Q table
        - lambtha: the eligibility trace factor
        - episodes: the total number of episodes to train over
        - max_steps: the maximum number of steps per episode
        - alpha: the learning rate
        - gamma: the discount rate
        - epsilon: the initial threshold for epsilon greedy
        - min_epsilon: the minimum value that epsilon should decay to
        - epsilon_decay: the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """

    # Initial value of epsilon for decay calculation
    epsilon_init = epsilon

    for ep in range(episodes):
        # Reset environment to start a new episode
        state = env.reset()[0]
        # Choose initial action using epsilon-greedy policy
        action = epsilon_greedy(Q, state, epsilon)
        # Initialize eligibility trace for Q table
        eligibility_trace = np.zeros_like(Q)

        for step in range(max_steps):
            # Take action in the environment and observe new state and reward
            new_state, reward, done, truncated, _ = env.step(action)
            # Choose new action using epsilon-greedy policy for new state
            new_action = epsilon_greedy(Q, new_state, epsilon)

            # Calculate TD error uning action-value form of Q
            td_error = (reward + gamma * Q[new_state, new_action]) \
                - Q[state, action]

            # Update eligibility trace for current state and action
            eligibility_trace[state, action] += 1
            eligibility_trace *= gamma * lambtha

            # Update Q table using SARSA(λ) update rule
            Q += alpha * td_error * eligibility_trace

            # Update state and action for next iteration
            state = new_state
            action = new_action

            if done or truncated:
                break  # Exit loop if episode is finished

        # Decay epsilon according to exponential schedule
        epsilon = min_epsilon + (epsilon_init - min_epsilon) * np.exp(
            -epsilon_decay * ep)

    # Return the updated Q table
    return Q
