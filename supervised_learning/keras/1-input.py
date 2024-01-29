#!/usr/bin/env python3
""" Function that builds a neural network with the Keras library """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that builds a neural network with the Keras library
    Arguments:
        - nx is the number of input features to the network
        - layers is a list containing the number of nodes in each layer of
            the network
        - activations is a list containing the activation functions used for
            each layer of the network
        - lambtha is the L2 regularization parameter
        - keep_prob is the probability that a node will be kept for dropout
    Returns:
        The keras model
    """
    # Define the input layer
    input_layer = K.Input(shape=(nx,))
    x = input_layer
    # Loop trhough the layers to build the neural network with dropout
    for i in range(len(layers)):
        # For each layer, add Dense layer with specified parameters
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)
        # Add dropout layer for regularization if it's not the last layer
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=input_layer, outputs=x)
    return model
