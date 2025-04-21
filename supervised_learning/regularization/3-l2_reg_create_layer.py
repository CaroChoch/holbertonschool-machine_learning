#!/usr/bin/env python3
"""Function that creates a tensorflow layer that includes L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function that creates a tensorflow layer that includes L2 regularization
    Arguments:
     - prev is a tensor containing the output of the previous layer
     - n is the number of nodes the new layer should contain
     - activation is the activation function that should be used on the layer
     - lambtha is the L2 regularization parameter
    Returns:
     The output of the new layer
    """
    # Initialize weights using He et al. initialization
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode="fan_avg")
    # Regularize the weights with L2 regularization
    regularizer = tf.keras.regularizers.l2(lambtha)
    # Create the layer with the previous initializer and regularizer
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)
    # Return the output of the new layer
    return layer(prev)
