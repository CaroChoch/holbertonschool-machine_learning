#!/usr/bin/env python3
"""
Function that calculates the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates the cost of a neural network with L2
    regularization
    Arguments:
     - cost is the cost of the network without L2 regularization
     - lambtha is the regularization parameter
     - weights is a dictionary of the weights and biases (numpy.ndarrays) of
        the neural network
     - L is the number of layers in the neural network
     - m is the number of data points used
     Returns:
        The cost of the network accounting for L2 regularization
    """
    """sum = 0
    for i in range(1, L + 1):
        sum = sum + np.linalg.norm(weights['W' + str(i)])
    return cost + (lambtha / (2 * m)) * sum"""
    # Initialize the sum of squared weights
    sum_weight = 0

    # Iterate through each layer
    for i in range(1, L + 1):
        # Retrieve the weights for the current layer
        weight = weights['W' + str(i)]

        # Add the sum of squared weights for the current layer
        sum_weight += np.sum(np.square(weight))

    # Calculate the L2 regularized cost by adding the regularization term
    l2_cost = cost + (lambtha / (2 * m)) * sum_weight

    # Return the final L2 regularized cost
    return l2_cost
