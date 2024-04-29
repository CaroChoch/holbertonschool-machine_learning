#!/usr/bin/env python3
""" RNN Cell """
import numpy as np


class RNNCell:
    """ RNNCell class """
    def __init__(self, i, h, o):
        """
        Class constructor
        Arguments:
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs
        """
        # Initialization of Weights of the cell for the concatenated hidden state and input data
        self.Wh = np.random.randn(h + i, h)
        # Initialization of Weights of the cell that will be used in the output
        self.Wy = np.random.randn(h, o)
        # Initialization of biases of the cell for the concatenated hidden state and input data
        self.bh = np.zeros((1, h))
        # Initialization of biases of the cell that will be used in the output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Method that performs forward propagation for one time step
        Arguments:
            - h_prev is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
                * m is the batch size for the data
            - x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
                * m is the batch size for the data
        Returns: h_next, y
            - h_next is the next hidden state
            - y is the output of the cell
        """
        # Concatenate the hidden state and the input data
        h_concatenate = np.concatenate((h_prev, x_t), axis=1)
        # Calculate the next hidden state
        h_next = np.tanh(np.dot(h_concatenate, self.Wh) + self.bh)
        # Calculate the output of the cell
        y = np.dot(h_next, self.Wy) + self.by
        # Apply the softmax activation function
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        # Return the next hidden state and the output of the cell
        return h_next, y
