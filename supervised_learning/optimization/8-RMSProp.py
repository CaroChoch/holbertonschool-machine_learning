#!/usr/bin/env python3
"""
Function that creates the training operation for a neural
network in tensorflow using the RMSProp optimization algorithm
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Function that creates the training operation for a neural
    network in tensorflow using the RMSProp optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta2 is the RMSProp weight
     - epsilon is a small number to avoid division by zero
    Returns:
     The RMSProp optimization operation
    """

    # Define the optimizer with RMSProp
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)

    # Create the RMSProp optimization operation
    RMSProp_op = optimizer.minimize(loss)

    # Return the RMSProp optimization operation
    return RMSProp_op
