#!/usr/bin/env python3
"""
Function that builds the DenseNet-121 architecture as described
in Densely Connected Convolutional Networks
"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Function that builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks
    Arguments:
        - growth_rate is the growth rate
        - compression is the compression factor
    Returns:
        the keras model
    """
    # Input layer and initializer
    inputs = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal(seed=0)

    # Initial Batch Normalization and Activation
    x = K.layers.BatchNormalization(axis=3)(inputs)
    x = K.layers.Activation('relu')(x)

    # Initial number of filters
    nb_filters = growth_rate * 2

    # Convolutional layer 1
    x = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=initializer)(x)

    # Max pool layer
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(x)

    # Dense Block 1 + Transition layer 1
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 2 + Transition layer 2
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 3 + Transition layer 3
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 4
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    # Average pool layer
    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  padding='same')(x)

    # Fully connected output
    x = K.layers.Dense(units=1000,
                       activation='softmax',
                       kernel_initializer=initializer)(x)

    model = K.models.Model(inputs=inputs, outputs=x)
    return model
