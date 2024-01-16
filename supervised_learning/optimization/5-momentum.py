#!/usr/bin/env python3
"""
Function that updates a variable using the gradient descent with
momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function that updates a variable using the gradient descent
    with momentum optimization algorithm
    Arguments:
     - alpha is the learning rate
     - beta1 is the momentum weight
     - var is a numpy.ndarray containing the variable to be updated
     - grad is a numpy.ndarray containing the gradient of var
     - v is the previous first moment of var
    Returns:
     The updated variable and the new moment, respectively
    """
    # Compute the exponentially weighted average of the gradient
    # Momentum formula:
    Vdw = (beta1 * v) + ((1 - beta1) * grad)
    # Update the variable using the momentum-based gradient descent
    W = var - (alpha * Vdw)
    # Return the updated variable and the new moment, respectively
    return W, Vdw
