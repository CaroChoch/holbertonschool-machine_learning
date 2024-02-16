#!/usr/bin/env python3
"""
Function that that builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)
"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function that that builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
        Returns: the keras Model
    """

    # define input & initializer
    inputs = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal()

    # conv1
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(inputs)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu_1 = K.layers.Activation(activation='relu')(batch_normalization_1)

    # max pool layer
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(relu_1)
    # conv2_x
    projection_block_1 = projection_block(pool1, [64, 64, 256], s=1)
    identity_block_1 = identity_block(projection_block_1, [64, 64, 256])
    identity_block_2 = identity_block(identity_block_1, [64, 64, 256])

    # conv3_x
    projection_block_2 = projection_block(
        identity_block_2,
        [128, 128, 512],
        s=2)
    identity_block_3 = identity_block(projection_block_2, [128, 128, 512])
    identity_block_4 = identity_block(identity_block_3, [128, 128, 512])
    identity_block_5 = identity_block(identity_block_4, [128, 128, 512])

    # conv4_x
    projection_block_3 = projection_block(
        identity_block_5,
        [256, 256, 1024],
        s=2)
    identity_block_6 = identity_block(projection_block_3, [256, 256, 1024])
    identity_block_7 = identity_block(identity_block_6, [256, 256, 1024])
    identity_block_8 = identity_block(identity_block_7, [256, 256, 1024])
    identity_block_9 = identity_block(identity_block_8, [256, 256, 1024])
    identity_block_10 = identity_block(identity_block_9, [256, 256, 1024])

    # conv5_x
    projection_block_4 = projection_block(
        identity_block_10,
        [512, 512, 2048],
        s=2)
    identity_block_11 = identity_block(projection_block_4, [512, 512, 2048])
    identity_block_12 = identity_block(identity_block_11, [512, 512, 2048])

    # Average pool layer
    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1),
                                      padding='valid')(identity_block_12)

    # Fully connected
    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(pool2)

    model = K.models.Model(inputs=inputs, outputs=output)

    return model
