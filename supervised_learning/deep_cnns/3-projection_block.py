#!/usr/bin/env python3
"""projection_block module"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in "Deep Residual Learning for Image Recognition" (2015).

    Arguments:
    A_prev -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- Python tuple or list of integers, defining the number of filters in the CONV layers of the main path
               (F11, F3, F12):
               F11 -- number of filters for the first 1x1 convolution
               F3  -- number of filters for the 3x3 convolution
               F12 -- number of filters for the second 1x1 convolution and the shortcut projection
    s -- Integer, specifying the stride to be used in the first convolution in the main path and the shortcut

    Returns:
    X -- output of the projection block, tensor of shape (m, n_H, n_W, n_C)
    """
    # Retrieve filters
    F11, F3, F12 = filters

    # He normal initializer with seed=0
    initializer = K.initializers.he_normal(seed=0)

    # Main path
    # First component: 1x1 conv
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        strides=(s, s),
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component: 3x3 conv
    X = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component: 1x1 conv
    X = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path (projection)
    shortcut = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add main path and shortcut
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
