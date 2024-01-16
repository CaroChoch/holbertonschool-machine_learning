#!/usr/bin/env python3
"""
Function that that updates a variable using the RMSProp
optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Function that that updates a variable using the RMSProp
    optimization algorithm
    Arguments:
     - alpha is the learning rate
     - beta2 is the RMSProp weight
     - epsilon is a small number to avoid division by zero
     - var is a numpy.ndarray containing the variable to be updated
     - grad is a numpy.ndarray containing the gradient of var
     - s is the previous second moment of var
    Returns:
     The updated variable and the new moment, respectively
    """
    # Compute the RMSProp
    Sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    # Update the variable using the RMSProp
    W = var - (alpha * (grad / ((Sdw ** (1/2)) + epsilon)))
    # Return the updated variable and the new moment, respectively
    return W, Sdw
