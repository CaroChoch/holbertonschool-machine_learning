#!/usr/bin/env python3
""" Function that conducts forward propagation using Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout
    Arguments:
     - X is a numpy.ndarray of shape (nx, m) containing the input data for
        the network
        * nx is the number of input features
        * m is the number of data points
     - weights is a dictionary of the weights and biases of the neural network
     - L the number of layers in the network
     - keep_prob is the probability that a node will be kept
     Returns:
      A dictionary containing the outputs of each layer and the dropout mask
      used on each layer
    """
    # Initialize the cache dictionary to store intermediate values
    cache = {}
    cache['A0'] = X  # Set the input as the first activation value

    # Loop through the layers of the neural network
    for i in range(L):
        W = weights['W' + str(i + 1)]  # Get the weights for the current layer
        b = weights['b' + str(i + 1)]  # Get the biases for the current layer
        A = cache['A' + str(i)]  # Get the activation for the previous layer
        Z = np.matmul(W, A) + b  # Calculate the linear function

        # If it is the last layer, perform softmax activation
        if i == L - 1:
            t = np.exp(Z)
            cache['A' + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
        # Otherwise, use the tanh activation function for the hidden layers
        else:
            cache['A' + str(i + 1)] = np.tanh(Z)
            # Apply dropout regularization to the hidden layers
            d = np.random.rand(cache['A' + str(i + 1)].shape[0],
                               cache['A' + str(i + 1)].shape[1])
            d = np.where(d < keep_prob, 1, 0)  # Apply the mask
            cache['D' + str(i + 1)] = d  # Store the mask
            cache['A' + str(i + 1)] *= d  # Apply the mask to the activation
            cache['A' + str(i + 1)] /= keep_prob  # Adjust the activation

    # Return the cache dictionary
    return cache
