#!/usr/bin/env python3
"""
Function that builds a projection block as described in Deep Residual
Learning for Image Recognition (2015).
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning
    for Image Recognition (2015).

    Args:
        A_prev: output from previous layer.
        filters: tuple or list containing F11, F3, F12.
        s: stride for the first convolution in main path and shortcut.

    Returns:
        Activated output of the projection block.
    """
    F11, F3, F12 = filters

    # He normal initialization est couramment utilisée pour ReLU
    initializer = K.initializers.he_normal(seed=0)

    # Premier composant du chemin principal
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(bn1)

    # Deuxième composant du chemin principal
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(bn2)

    # Troisième composant du chemin principal
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Connexion de raccourci
    sc = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn_sc = K.layers.BatchNormalization(axis=3)(sc)

    # Addition du chemin principal et du raccourci
    add = K.layers.Add()([bn3, bn_sc])

    # Activation finale
    out = K.layers.Activation('relu')(add)

    return out
