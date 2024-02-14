#!/usr/bin/env python3
"""
Function that builds the inception network as described in Going
Deeper with Convolutions (2014)
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds the inception network as described in Going
    Deeper with Convolutions (2014)
        Returns : the keras model
    """
    # Define Input data
    input_data = K.Input(shape=(224, 224, 3))

    # Initial convolution with a large kernel and a stride of 2
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        activation='relu',
        strides=(2, 2),
        padding='same')(input_data)

    # Max pooling to reduce spatial dimensions
    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(conv1)

    # 1x1 convolution followed by 3x3 convolution
    conv2 = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation='relu',
        strides=(1, 1),
        padding='same')(pool1)

    conv3 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        activation='relu',
        strides=(1, 1),
        padding='same')(conv2)

    # Max pooling to reduce spatial dimensions
    pool2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(conv3)

    # Inception blocks 3a and 3b
    inception3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])

    # Max pooling to reduce spatial dimensions
    pool3 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(inception3b)

    # Inception blocks 4a, 4b, 4c, 4d and 4e
    inception4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])

    # Max pooling to reduce spatial dimensions
    pool4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(inception4e)

    # Inception blocks 5a and 5b
    inception5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])

    # Average pooling to reduce spatial dimensions
    pool5 = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1))(inception5b)

    # Dropout for regularization
    dropout = K.layers.Dropout(rate=0.4)(pool5)

    # Fully connected layer with softmax activation for classification
    softmax = K.layers.Dense(
        units=1000,
        activation='softmax')(dropout)

    model = K.models.Model(inputs=input_data, outputs=softmax)

    return model
