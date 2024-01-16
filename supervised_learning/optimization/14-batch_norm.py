#!/usr/bin/env python3
"""
Function that creates a batch normalization layer for a neural
network in tensorflow
"""
import tensorflow.compat.v1 as tf


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
                            kernel_initializer=init,
                            name="layer")

    # Generate the output of the layer
    Z = layer(prev)

    # Calculate the mean and variance of Z
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Gamma and Beta initialization parameters
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")

    # Epsilon value to avoid division by zero
    epsilon = 1e-8

    # Normalize the output of the layer
    Z_norm = tf.nn.batch_normalization(x=Z,
                                       mean=mean,
                                       variance=variance,
                                       offset=beta,
                                       scale=gamma,
                                       variance_epsilon=epsilon)

    # Return the activation function applied to Z
    return activation(Z_norm)
