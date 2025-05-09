#!/usr/bin/env python3
"""
Function that builds a transition layer as described in Densely
Connected Convolutional Networks
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as in DenseNet with compression (DenseNet-C).

    Arguments:
    - X: Input tensor from previous layer
    - nb_filters: Integer, number of filters in X
    - compression: Float, compression factor for number of filters

    Returns:
    - output tensor after transition layer
    - updated number of filters
    """
    # He normal initializer with fixed seed for reproducibility
    initializer = K.initializers.he_normal(seed=0)

    # Compute compressed number of filters
    nb_filters_after_compression = int(nb_filters * compression)

    # BatchNormalization -> ReLU activation
    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.ReLU()(x)

    # 1x1 convolution to reduce feature-map dimensions
    x = K.layers.Conv2D(
        filters=nb_filters_after_compression,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)

    # Average pooling to halve spatial dimensions
    x = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )(x)

    return x, nb_filters_after_compression
