#!/usr/bin/env python3
"""
DenseNet-121 implementation following Huang et al. (2017).

Constructs a densely connected convolutional network with 121 layers,
including four dense blocks and three transition layers for downsampling.
Reference:
    Gao Huang, et al. "Densely Connected Convolutional Networks", CVPR 2017.
"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 model.

    Args:
        growth_rate (int): Feature maps added per layer in each dense block.
        compression (float): Factor (<=1) to reduce channels in transition layers.

    Returns:
        Keras Model: Instance of DenseNet-121 ready for training or inference.
    """
    # Input: 224x224 RGB images
    inputs = K.layers.Input(shape=(224, 224, 3))

    # Initial batch norm + ReLU activation
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.ReLU()(x)

    # Initial convolution: 7x7 kernel, stride 2, filters = 2 * growth_rate
    num_filters = growth_rate * 2
    x = K.layers.Conv2D(
        filters=num_filters,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    # Downsample spatial dimensions: 3x3 max pooling, stride 2
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Add dense blocks with transition layers in between
    x, num_filters = dense_block(x, num_filters, growth_rate, 6)
    x, num_filters = transition_layer(x, num_filters, compression)
    x, num_filters = dense_block(x, num_filters, growth_rate, 12)
    x, num_filters = transition_layer(x, num_filters, compression)
    x, num_filters = dense_block(x, num_filters, growth_rate, 24)
    x, num_filters = transition_layer(x, num_filters, compression)
    x, num_filters = dense_block(x, num_filters, growth_rate, 16)

    # Global average pooling to create feature vector
    x = K.layers.AveragePooling2D(pool_size=7, padding='same')(x)

    # Classification layer: 1000 classes with softmax activation
    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    # Instantiate model
    return K.models.Model(inputs=inputs, outputs=outputs)
