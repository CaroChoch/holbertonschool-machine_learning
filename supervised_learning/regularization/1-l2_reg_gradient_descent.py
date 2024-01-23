#!/usr/bin/env python3
"""
Function that updates the weights and biases of a neural
network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a neural
    network using gradient descent with L2 regularization
    Arguments:
     - Y is a one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
        * classes is the number of classes
        * m is the number of data points
     - weights is a dictionary of the weights and biases of the neural network
     - cache is a dictionary of the outputs of each layer of the neural network
     - alpha is the learning rate
     - lambtha is the L2 regularization parameter
     - L is the number of layers of the network
    """
    # Initialize the number of layers of the network
    m = Y.shape[1]

    # Calculate the derivative of the loss function with respect to z
    dz = cache['A' + str(L)] - Y

    # Iterate through the layers backwards
    for i in range(L, 0, -1):
        # Calculate the L2 regularization term
        l2_reg_term = (lambtha / m) * weights['W' + str(i)]

        # Calculate the derivative of the weights (with L2 regularization)
        dw = np.matmul(dz, cache['A' + str(i - 1)].T) / m + l2_reg_term

        # Calculate the derivative of the biases
        db = np.sum(dz, axis=1, keepdims=True) / m

        # Update the weights and biases using gradient descent
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db

        # If it is not the output layer, calculate the derivative of the
        # the activation function (tanh) and multiply it by the derivative
        # of the loss function with respect to z
        if i == 1:
            # Calculate the derivative of tanh activation
            tanh_derivative = 1 - np.power(cache["A" + str(i)], 2)
            # Update dz for tanh activation
            dz = np.matmul(dz, tanh_derivative.T)
        else:
            # Update dz for other activation functions
            dz = np.matmul(weights['W' + str(i)].T, dz)
