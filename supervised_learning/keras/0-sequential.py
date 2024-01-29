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
    # Create a sequential model
    model = K.Sequential()
    # Loop trhough the layers to build the neural network with dropout
    for i in range(len(layers)):
        if i == 0:
            # For the first layer, specify the input shape and add L2 reg
            model.add(K.layers.Dense(
                layers[i],
                input_shape=(nx,),
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))
        else:
            # For the rest of the layers, just specify the number of nodes and
            # add L2
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))
        # Add dropout layer for regularization if it's not the last layer
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
