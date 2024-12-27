#!/usr/bin/env python3
""" Creates the forward propagation graph for the neural network """

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural network

    Arguments:
     - x is the placeholder for the input data
     - layer_sizes is a list containing the number of nodes in each layer
        of the network
     - activations is a list containing the activation functions for each
        layer of the network

    Returns:
     The prediction of the network in tensor form
    """
    # Create the first layer
    pred = x

    # Create the subsequent layers
    for size, activ in zip(layer_sizes, activations):
        pred = create_layer(pred, size, activ)

    # Return the prediction
    return pred
