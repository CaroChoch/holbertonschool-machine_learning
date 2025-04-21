#!/usr/bin/env python3
"""
Function that builds a modified version of the LeNet-5 architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5 architecture
    using Keras
    Arguments:
        - X is a K.Input of shape (m, 28, 28, 1) containing the input images
        for the network
            * m is the number of images
    Returns:
        a K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics
    """

    initializer = K.initializers.HeNormal(seed=None)

    # Layer 1: Convolutional layer with 6 kernels of shape 5x5 and same padding
    convolutional_layer1 = K.layers.Conv2D(filters=6,
                                           kernel_size=(5, 5),
                                           padding='same',
                                           activation='relu',
                                           kernel_initializer=initializer)(X)

    # Layer 2: Max pooling layer with kernels of shape 2x2 and 2x2 strides
    max_pool_layer1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                            strides=(2, 2))(
                                                convolutional_layer1)

    # Layer 3: Convolutional layer with 16 kernels of shape 5x5 and valid pad
    convolutional_layer2 = K.layers.Conv2D(filters=16,
                                           kernel_size=(5, 5),
                                           padding='valid',
                                           activation='relu',
                                           kernel_initializer=initializer)(
                                               max_pool_layer1)

    # Layer 4: Max pooling layer with kernels of shape 2x2 and 2x2 strides
    max_pool_layer2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                            strides=(2, 2))(
                                                convolutional_layer2)

    # Flatten the output for fully connected layers
    flattened = K.layers.Flatten()(max_pool_layer2)

    # Layer 5: Fully connected layer with 120 nodes
    fully_connected_layer1 = K.layers.Dense(units=120,
                                            activation='relu',
                                            kernel_initializer=initializer)(
                                                flattened)

    # Layer 6: Fully connected layer with 84 nodes
    fully_connected_layer2 = K.layers.Dense(units=84,
                                            activation='relu',
                                            kernel_initializer=initializer)(
                                                fully_connected_layer1)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    output_layer = K.layers.Dense(units=10,
                                  activation='softmax',
                                  kernel_initializer=initializer)(
                                      fully_connected_layer2)

    # Create the model
    model = K.Model(inputs=X, outputs=output_layer)

    # Compile the model using Adam optimization and accuracy metric
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
