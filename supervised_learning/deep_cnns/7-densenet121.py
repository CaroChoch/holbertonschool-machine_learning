#!/usr/bin/env python3
"""
DenseNet-121 architecture implementation following Huang et al. (2017).

This model consists of four dense blocks, each separated by a transition layer,
allowing efficient feature reuse and parameter reduction via compression.

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
        growth_rate (int): Number of feature maps to add per layer in each dense block.
        compression (float): Compression factor (<=1.0) to reduce channels in transition layers.

    Returns:
        keras.Model: Constructed DenseNet-121 model ready for training or inference.
    """
    # Input tensor for 224x224 RGB images
    inputs = K.layers.Input(shape=(224, 224, 3))

    # Initial batch normalization and ReLU activation
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Activation('relu')(x)

    # 7x7 convolution with stride 2; initialize filters to twice the growth rate
    num_filters = growth_rate * 2
    x = K.layers.Conv2D(
        filters=num_filters,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    # 3x3 max pooling with stride 2 to reduce spatial resolution
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Dense block 1: 6 layers, followed by a transition layer
    x, num_filters = dense_block(x, num_filters, growth_rate, 6)
    x, num_filters = transition_layer(x, num_filters, compression)

    # Dense block 2: 12 layers, followed by a transition layer
    x, num_filters = dense_block(x, num_filters, growth_rate, 12)
    x, num_filters = transition_layer(x, num_filters, compression)

    # Dense block 3: 24 layers, followed by a transition layer
    x, num_filters = dense_block(x, num_filters, growth_rate, 24)
    x, num_filters = transition_layer(x, num_filters, compression)

    # Dense block 4: 16 layers, no transition afterward
    x, num_filters = dense_block(x, num_filters, growth_rate, 16)

    # Global average pooling collapses spatial dimensions to 1x1 per feature map
    x = K.layers.AveragePooling2D(pool_size=7, padding='same')(x)

    # Final classification layer: 1000-way softmax
    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    # Create and return the model
    return K.models.Model(inputs=inputs, outputs=outputs)
