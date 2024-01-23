#!/usr/bin/env python3
""" Function that creates a layer of a neural network using dropout """
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Function that creates a layer of a neural network using dropout
    Arguments:
     - prev is a tensor containing the output of the previous layer
     - n is the number of nodes the new layer should contain
     - activation is the activation function that should be used on the layer
     - keep_prob is the probability that a node will be kept
    Returns:
     The output of the new layer
    """
    # Initialize weights using He et al. initialization
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode="fan_avg")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name="layer")
    # Apply dropout to the layer
    dropout = tf.layers.Dropout(keep_prob)
    # Return the output of the new layer
    return dropout(layer(prev))
