#!/usr/bin/env python3
"""
Function that builds a transition layer as described in Densely
Connected Convolutional Networks
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer as described in Densely
    Connected Convolutional Networks

    Arguments:
        - X is the output from the previous layer
        - nb_filters is an integer representing the number of filters in X
        - compression is the compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters within
        the output, respectively
    """
    # He normal initialization is commonly used for ReLU activation functions
    initializer = K.initializers.he_normal(seed=None)

    # Adjust the number of filters based on the compression factor
    nb_filters_after_compression = int(nb_filters * compression)

    # Batch normalization to normalize the inputs
    batch_normalization = K.layers.BatchNormalization(axis=3)(X)

    # ReLU activation for introducing non-linearity
    activated_output = K.layers.Activation('relu')(batch_normalization)

    # 1x1 Convolution layer to adjust the number of filters
    conv_1x1 = K.layers.Conv2D(
            filters=nb_filters_after_compression,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(activated_output)

    # Average pooling to reduce the spatial dimensions
    average_pooling = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )(conv_1x1)

    # Return output of the transition layer and the updated number of filters
    return average_pooling, nb_filters_after_compression
