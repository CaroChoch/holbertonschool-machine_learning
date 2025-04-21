#!/usr/bin/env python3
"""
Function that creates a batch normalization layer for a neural
network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that creates a batch normalization layer for a neural
    network in tensorflow
    Arguments:
     - prev is the activated output of the previous layer
     - n is the number of nodes in the layer to be created
     - activation is the activation function that should be used
        on the output of the layer
    Returns:
     A tensor of the activated output for the layer
    """
    # Initialize the weights and biases of the layer
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.keras.layers.Dense(units=n,
                                  kernel_initializer=init)

    # Generate the output of the layer
    Z = layer(prev)

    # Calculate the mean and variance of Z
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Gamma and Beta initialization parameters
    gamma = tf.Variable(initial_value=tf.ones([n]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([n]), trainable=True)

    # Epsilon value to avoid division by zero
    epsilon = 1e-7

    # Normalize the output of the layer
    Z_norm = tf.nn.batch_normalization(Z,
                                       mean,
                                       variance,
                                       beta,
                                       gamma,
                                       epsilon)

    # Return the activation function applied to Z
    return activation(Z_norm)
