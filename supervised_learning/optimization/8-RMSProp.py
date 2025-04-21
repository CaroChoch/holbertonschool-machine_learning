#!/usr/bin/env python3
"""
Function that creates the training operation for a neural
network in tensorflow using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Function that creates the training operation for a neural
    network in tensorflow using the RMSProp optimization algorithm
    Arguments:
     - alpha is the learning rate
     - beta2 is the RMSProp weight
     - epsilon is a small number to avoid division by zero
    Returns:
     The RMSProp optimizer
    """

    # Define the optimizer with RMSProp
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2,
                                            epsilon=epsilon)

    # Return the RMSProp optimization operation
    return optimizer
