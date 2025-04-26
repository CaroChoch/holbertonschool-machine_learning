#!/usr/bin/env python3
"""
Function that builds an inception block as described in Going Deeper
with Convolutions (2014)
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Function that builds an inception block as described in Going Deeper
    with Convolutions (2014)
    Arguments:
        - A_prev is the output from the previous layer
        - filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
        respectively:
            * F1 is the number of filters in the 1x1 convolution
            * F3R is the number of filters in the 1x1 convolution before the
            3x3 convolution
            * F3 is the number of filters in the 3x3 convolution
            * F5R is the number of filters in the 1x1 convolution before the
            5x5 convolution
            * F5 is the number of filters in the 5x5 convolution
            * FPP is the number of filters in the 1x1 convolution after the
            max pooling
    Returns:
        The concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    initializer = K.initializers.HeNormal(seed=None)

    # Convolutional layer with F1 kernels of shape 1x1 and same padding
    layer_F1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=initializer)(A_prev)

    # Convolutional layer with F3R kernels of shape 1x1 and same padding
    layer_F3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=initializer)(A_prev)

    # Convolutional layer with F3 kernels of shape 3x3 and same padding
    layer_F3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=initializer)(layer_F3R)

    # Convolutional layer with F5R kernels of shape 1x1 and same padding
    layer_F5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=initializer)(A_prev)

    # Convolutional layer with F5 kernels of shape 5x5 and same padding
    layer_F5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer)(layer_F5R)

    # Max pooling layer with kernels of shape 3x3 and 1x1 strides
    layer_max_pooling = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    # Convolutional layer with FPP kernels of shape 1x1 and same padding
    layer_FPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(layer_max_pooling)

    # concatenate the output
    model = K.layers.concatenate([layer_F1, layer_F3, layer_F5, layer_FPP])

    return model
