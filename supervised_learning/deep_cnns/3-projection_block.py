#!/usr/bin/env python3
"""
Module that defines a projection block for ResNet
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in "Deep Residual
    Learning for Image Recognition (2015)"

    Arguments:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12, respectively
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution
                 and the 1x1 convolution in the shortcut connection
        s: stride of the first convolution in both the main path
           and the shortcut connection

    Returns:
        The activated output of the projection block
    """
    # Get filter values
    F11, F3, F12 = filters

    # Save the input value for the shortcut connection
    X_shortcut = A_prev

    # Main path
    # First component - 1x1 convolution with s stride
    X = K.layers.Conv2D(filters=F11,
                         kernel_size=1,
                         strides=s,
                         padding='same',
                         kernel_initializer=K.initializers.he_normal(seed=0))(A_prev)
    
    # Batch normalization
    X = K.layers.BatchNormalization(axis=3)(X)
    
    # ReLU activation
    X = K.layers.Activation('relu')(X)

    # Second component - 3x3 convolution
    X = K.layers.Conv2D(filters=F3,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer=K.initializers.he_normal(seed=0))(X)
    
    # Batch normalization
    X = K.layers.BatchNormalization(axis=3)(X)
    
    # ReLU activation
    X = K.layers.Activation('relu')(X)

    # Third component - 1x1 convolution
    X = K.layers.Conv2D(filters=F12,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer=K.initializers.he_normal(seed=0))(X)
    
    # Batch normalization
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path - 1x1 convolution with s stride
    X_shortcut = K.layers.Conv2D(filters=F12,
                                  kernel_size=1,
                                  strides=s,
                                  padding='same',
                                  kernel_initializer=K.initializers.he_normal(seed=0))(X_shortcut)
    
    # Batch normalization
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add the shortcut to the main path
    X = K.layers.Add()([X, X_shortcut])
    
    # Final ReLU activation
    X = K.layers.Activation('relu')(X)

    return X
