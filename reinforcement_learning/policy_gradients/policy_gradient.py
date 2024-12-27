#!/usr/bin/env python3
""" computes to policy with a weight of a matrix """
import numpy as np


def policy(matrix, weight):
    """
    Computes to policy with a weight of a matrix
    Arguments:
        - matrix is a state representing the current observation of the env
        - weight is a matrix of random weight
    Returns:
        the action and the gradient (in this order)
    """
    # Compute the dot product of the state matrix and the weight matrix
    z = matrix.dot(weight)

    # Compute the exponential of each element in z (softmax numerator)
    exp = np.exp(z)

    # Compute the softmax probabilities by normalizing exp by its sum
    policy = exp / np.sum(exp)

    return policy


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a weight
    Arguments:
        - state: matrix representing the current observation of the environment
        - weight: matrix of random weight
    Returns:
        the action and the gradient (in this order)
    """
    # Compute policy (action probabilities) using the current state and weight
    P = policy(state, weight)

    # Choose an action based on the computed probabilities
    action = np.random.choice(len(P[0]), p=P[0])

    # Initialize gradient of the policy as a copy of the policy probabilities
    policy_gradient_update = P.copy()

    # Adjust the gradient for the chosen action by subtracting 1
    policy_gradient_update[0, action] -= 1

    # Compute the final gradient by taking the dot product of the state
    # transpose and the policy gradient update
    gradient = state.T.dot(policy_gradient_update)

    return action, gradient
