#!/usr/bin/env python3
import numpy as np

def policy(state, weight):
    """
    Compute the action probabilities (softmax) given a
    state and weight matrix.
    Args:
        state: np.ndarray of shape (4,), the current observation.
        weight: np.ndarray of shape (4,2), the policy weights.
    Returns:
        probs: np.ndarray of shape (2,), probability of each action.
    """
    z = state.dot(weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient for a single step.
    Args:
        state: np.ndarray of shape (4,), the current observation.
        weight: np.ndarray of shape (4,2), the policy weights.
    Returns:
        action: int, the chosen action.
        gradient: np.ndarray of shape (4,2), gradient of log-probability.
    """
    # 1) compute action probabilities
    probs = policy(state, weight)
    # 2) sample an action according to the distribution
    action = np.random.choice(len(probs), p=probs)
    # 3) create one-hot vector for the chosen action
    one_hot = np.zeros_like(probs)
    one_hot[action] = 1
    # 4) compute gradient of log π(a|s): s ⊗ (one_hot - probs)
    grad = outer = np.outer(state, one_hot - probs)
    return action, grad
