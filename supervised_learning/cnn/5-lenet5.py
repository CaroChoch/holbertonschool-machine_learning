#!/usr/bin/env python3
"""Function that builds a modified version of the LeNet-5 architecture"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.
    Arguments:
        - X: K.Input of shape (m, 28, 28, 1)
    Returns:
        - A K.Model compiled with Adam optimizer and accuracy metric
    """

    def initializer():
        return K.initializers.HeNormal(seed=0)

    # 1. Conv layer: 6 filters, 5x5, same padding, relu
    x = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                        activation='relu', kernel_initializer=initializer())(X)
    # 2. Max pooling: 2x2, stride 2
    x = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 3. Conv layer: 16 filters, 5x5, valid padding, relu
    x = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                        activation='relu', kernel_initializer=initializer())(x)
    # 4. Max pooling: 2x2, stride 2
    x = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 5. Flatten
    x = K.layers.Flatten()(x)

    # 6. Fully connected: 120 nodes, relu
    x = K.layers.Dense(units=120, activation='relu',
                       kernel_initializer=initializer())(x)
    # 7. Fully connected: 84 nodes, relu
    x = K.layers.Dense(units=84, activation='relu',
                       kernel_initializer=initializer())(x)

    # 8. Output layer: 10 nodes, softmax
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=initializer())(x)

    # Model creation
    model = K.Model(inputs=X, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
