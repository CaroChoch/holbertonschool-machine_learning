#!/usr/bin/env python3
"""
Function that builds a dense block as described in Densely
Connected Convolutional Networks
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block as described in Densely
    Connected Convolutional Networks
    Arguments:
        - X is the output from the previous layer
        - nb_filters is an integer representing the number of filters in X
        - growth_rate is the growth rate for the dense block
        - layers is the number of layers in the dense block
    Returns:
        The concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs, respectively
    """
    # He normal initialization is commonly used for ReLU activation functions
    initializer = K.initializers.HeNormal(seed=None)

    # Loop through the specified number of layers in the dense block
    for layer in range(layers):

        # Batch normalization to normalize the inputs
        batch_normalization_1 = K.layers.BatchNormalization(axis=3)(X)

        # ReLU activation for introducing non-linearity
        activated_output_1 = K.layers.Activation('relu')(batch_normalization_1)

        # step 1
        # 1x1 Convolution Layer (Dimension Reduction):
        conv_1x1 = K.layers.Conv2D(
            filters=growth_rate * 4,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(activated_output_1)

        # Batch normalization after the first convolution
        batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv_1x1)

        # ReLU activation after the second batch normalization
        activated_output_2 = K.layers.Activation('relu')(batch_normalization_2)

        # step 2
        # 3x3 Convolution layer (Feature Extraction)
        # on reduced-dimensional representation
        conv_3x3 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=initializer
        )(activated_output_2)

        # Step 3
        # Concatenate previous layer's output with 3x3 convolution's output
        X = K.layers.concatenate([X, conv_3x3])

        # Update the total number of filters
        nb_filters += growth_rate

    # Returns the concatenated output and the updated number of filters
    return X, nb_filters
