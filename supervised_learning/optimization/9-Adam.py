#!/usr/bin/env python3
"""
Function that updates a variable in place using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Function that updates a variable in place using the Adam optimization
    algorithm
    Arguments:
     - alpha is the learning rate
     - beta1 is the weight used for the first moment
     - beta2 is the weight used for the second moment
     - epsilon is a small number to avoid division by zero
     - var is a numpy.ndarray containing the variable to be updated
     - grad is a numpy.ndarray containing the gradient of var
     - v is the previous first moment of var
     - s is the previous second moment of var
     - t is the time step used for bias correction
    Returns:
     The updated variable, the new first moment, and the new second moment,
     respectively
    """
    # Compute the first moment of the gradient
    Vdw = (beta1 * v) + ((1 - beta1) * grad)
    # Compute the second moment of the gradient
    Sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    # Compute bias-corrected first moment estimate
    Vdwc = Vdw / (1 - (beta1 ** t))
    # Compute bias-corrected second moment estimate
    Sdwc = Sdw / (1 - (beta2 ** t))
    # Update the variable
    W = var - (alpha * (Vdwc / ((Sdwc ** (1/2)) + epsilon)))
    # Return the updated variable, the new first moment,
    # and the new second moment, respectively
    return W, Vdw, Sdw
