#!/usr/bin/env python3
"""
Function that builds the DenseNet-121 architecture as described
in Densely Connected Convolutional Networks
"""

import tensorflow.keras as K
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
    initializer = K.initializers.HeNormal()

    # Initial Batch Normalization and ReLU activation
    batch_normalization = K.layers.BatchNormalization(axis=3)(inputs)
    relu = K.layers.Activation(activation='relu')(batch_normalization)

    # Initial number of filters
    nb_filters = int(growth_rate * 2)

    # Convolutional layer 1
    conv1 = K.layers.Conv2D(filters=nb_filters,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(relu)

    # max pool layer
    max_pool_layer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same')(conv1)

    # Dense Block 1 and Transition layer 1
    dense_block1, nb_filters = dense_block(max_pool_layer,
                                           nb_filters,
                                           growth_rate,
                                           6)
    transition_layer1, nb_filters = transition_layer(dense_block1,
                                                     nb_filters,
                                                     compression)

    # Dense Block 2 and Transition layer 2
    dense_block2, nb_filters = dense_block(transition_layer1,
                                           nb_filters,
                                           growth_rate,
                                           12)
    transition_layer2, nb_filters = transition_layer(dense_block2,
                                                     nb_filters,
                                                     compression)

    # Dense Block 3 and Transition layer 3
    dense_block3, nb_filters = dense_block(transition_layer2,
                                           nb_filters,
                                           growth_rate,
                                           24)
    transition_layer3, nb_filters = transition_layer(dense_block3,
                                                     nb_filters,
                                                     compression)

    # Dense Block 4
    dense_block4, nb_filters = dense_block(transition_layer3,
                                           nb_filters,
                                           growth_rate,
                                           16)

    # Average pool layer
    avg_pool_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                               strides=None,
                                               padding='same')(dense_block4)

    # Fully connected
    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(avg_pool_layer)

    model = K.models.Model(inputs=inputs, outputs=output)

    return model
