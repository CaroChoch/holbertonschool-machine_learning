#!/usr/bin/env python3
"""
Function that builds an identity block as described in Deep Residual
Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015)
        Arguments:
        - A_prev: is the output from the previous layer
        - filters: is a tuple or list containing F11, F3, F12, respectively:
            * F11: number of filters in the first 1x1 convolution
            * F3: number of filters in the 3x3 convolution
            * F12: number of filters in the second 1x1 convolution
        Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters

    # He normal initialization is commonly used for ReLU activations
    initializer = K.initializers.HeNormal(seed=None)

    # First component of main path
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu_1 = K.layers.Activation('relu')(batch_normalization_1)

    # Second component of main path
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(relu_1)
    batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu_2 = K.layers.Activation('relu')(batch_normalization_2)

    # Third component of main path
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu_2)
    batch_normalization_3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add the shortcut value to the output
    add = K.layers.Add()([batch_normalization_3, A_prev])

    # Activate the final output
    activated_output = K.layers.Activation('relu')(add)
    return activated_output
