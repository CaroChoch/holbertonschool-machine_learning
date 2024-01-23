#!/usr/bin/env python3
"""
Function that that updates the weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that that updates the weights of a neural network with Dropout
    regularization using gradient descent
    Arguments:
     - Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
        * classes is the number of classes
        * m is the number of data points
     - weights is a dictionary of the weights and biases of the neural network
     - cache is a dictionary of the outputs and dropout masks of each layer of
        the neural network
     - alpha is the learning rate
     - keep_prob is the probability that a node will be kept
     - L is the number of layers of the network
    """
    # Initialize the number of layers of the network
    m = Y.shape[1]

    # Calculate the derivative of the loss function with respect to z
    dz = cache['A' + str(L)] - Y

    # Iterate through the layers backwards
    for i in range(L, 0, -1):
        # Calculate the derivative of the weights
        dw = np.matmul(dz, cache['A' + str(i - 1)].T) / m

        # Calculate the derivative of the biases
        db = np.sum(dz, axis=1, keepdims=True) / m

        # If it is not the output layer, update dz for tanh activation
        # Calculate the derivative of tanh activation and multiply by dz
        dz = np.matmul(weights['W' + str(i)].T, dz) * (
            1 - cache["A" + str(i - 1)] ** 2)
        # If it is not the input layer, apply dropout regularization to the
        # hidden layers
        if i > 1:
            # Apply dropout regularization to the hidden layers
            dz *= cache['D' + str(i - 1)] / keep_prob

        # Update the weights and biases using gradient descent
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
