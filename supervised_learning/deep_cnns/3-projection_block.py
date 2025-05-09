#!/usr/bin/env python3
"""
Function that builds a projection block as described in Deep Residual Learning
for Image Recognition (2015)
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Function that builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015)
        Arguments:
        - A_prev: the output from the previous layer
        - filters: a tuple or list containing F11, F3, F12, respectively:
            * F11: the number of filters in the first 1x1 convolution
            * F3: the number of filters in the 3x3 convolution
            * F12: the number of filters in the second 1x1 convolution as well
            as the 1x1 convolution in the shortcut connection
        - s: the stride of the first convolution in both the main path and the
        shortcut connection
        Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters

    # He normal initialization est couramment utilisée pour ReLU
    initializer = K.initializers.HeNormal(seed=0)

    # Premier composant du chemin principal
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu_1 = K.layers.Activation('relu')(batch_normalization_1)

    # Deuxième composant du chemin principal
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu_1)
    batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu_2 = K.layers.Activation('relu')(batch_normalization_2)

    # Troisième composant du chemin principal
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu_2)
    batch_normalization_3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Connexion de raccourci
    shortcut_connection = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization_shortcut = K.layers.BatchNormalization(axis=3)(shortcut_connection)

    # Addition du chemin principal et du raccourci
    sum_result = K.layers.Add()(
        [batch_normalization_3, batch_normalization_shortcut]
    )

    # Activation finale
    activated_output = K.layers.Activation('relu')(sum_result)

    return activated_output
