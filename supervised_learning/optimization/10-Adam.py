#!/usr/bin/env python3
"""
Function that creates the training operation for a neural network in tensorflow
using the Adam optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    Arguments:
     - alpha is the learning rate
     - beta1 is the weight used for the first moment
     - beta2 is the weight used for the second moment
     - epsilon is a small number to avoid division by zero
    Returns:
     The Adam optimizer
    """

    # Define the optimizer with Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,
                                        beta1=beta1,
                                        beta2=beta2,
                                        epsilon=epsilon)

    # Return the Adam optimization operation
    return optimizer
