#!/usr/bin/env python3
""" Function that creates a layer """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Function that creates a layer

    Arguments:
     - prev is the tensor output of the previous layer
     - n is the number of nodes in the layer to create
     - activation is the activation function that the layer should use

    Returns:
     The tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer")
    return layer(prev)
