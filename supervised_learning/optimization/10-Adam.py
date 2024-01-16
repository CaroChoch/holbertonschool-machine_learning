#!/usr/bin/env python3
"""
Function that creates the training operation for a neural network in tensorflow
using the Adam optimization algorithm
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the weight used for the first moment
     - beta2 is the weight used for the second moment
     - epsilon is a small number to avoid division by zero
    Returns:
     The Adam optimization operation
    """

    # Define the optimizer with Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    # Create the Adam optimization operation
    Adam_op = optimizer.minimize(loss)

    # Return the Adam optimization operation
    return Adam_op
