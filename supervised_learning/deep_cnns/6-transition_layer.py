#!/usr/bin/env python3
"""
Function that builds a dense block as described in Densely
Connected Convolutional Networks
"""

from tensorflow import keras as K


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
    # Initialisation He normal avec graine fixe pour reproductibilité
    initializer = K.initializers.he_normal(seed=0)

    # Boucle sur les couches du dense block
    for _ in range(layers):
        # Bottleneck : BN -> ReLU -> Conv1x1
        bn1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=growth_rate * 4,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(act1)

        # BN -> ReLU -> Conv3x3
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=initializer
        )(act2)

        # Concatenation avec la sortie précédente
        X = K.layers.Concatenate(axis=3)([X, conv2])

        # Mise à jour du nombre de filtres total
        nb_filters += growth_rate

    return X, nb_filters
