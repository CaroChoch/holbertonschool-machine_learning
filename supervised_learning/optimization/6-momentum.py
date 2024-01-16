#!/usr/bin/env python3
"""
Function creates the training operation for a neural network
in tensorflow using the gradient descent with momentum optimization algorithm
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Function creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the momentum weight
    Returns:
     The momentum optimization operation
    """

    # Define the optimizer with momentum
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)

    # Create the momentum optimization operation
    momentum_op = optimizer.minimize(loss)

    # Return the momentum optimization operation
    return momentum_op
