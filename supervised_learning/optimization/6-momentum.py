#!/usr/bin/env python3
"""
Function creates the training operation for a neural network
in tensorflow using the gradient descent with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Create the optimizer using momentum for gradient descent
    Arguments:
     - alpha is the learning rate
     - beta1 is the momentum weight
    Returns:
     The momentum optimizer
    """

    # Define the optimizer with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    # Return the momentum optimizater
    return optimizer
