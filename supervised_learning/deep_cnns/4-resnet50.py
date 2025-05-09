#!/usr/bin/env python3
"""
Function that builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)
"""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).
    Returns: the Keras Model
    """
    # Input & initializer
    inputs = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(seed=0)

    # conv1
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=initializer)(inputs)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # conv3_x
    x = projection_block(x, [128, 128, 512], s=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # conv4_x
    x = projection_block(x, [256, 256, 1024], s=2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    # conv5_x
    x = projection_block(x, [512, 512, 2048], s=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # Average pool & output
    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=initializer)(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
