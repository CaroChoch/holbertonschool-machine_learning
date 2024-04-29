#!/usr/bin/env python3
""" Bidirectional cell Forward of an RNN """
import numpy as np


class BidirectionalCell:
    """ Bidirectional cell of a simple RNN """
    def __init__(self, i, h, o):
        """
        Class constructor
        Arguments:
         - i is the dimensionality of the data
         - h is the dimensionality of the hidden states
         - o is the dimensionality of the outputs
         - Public instance attributes:
            * Whf and bhf are for the hidden states in the forward direction
            * Whb and bhb are for the hidden states in the backward direction
            * Wy and by are for the outputs
        """
        # Initialize weights for the forward direction
        self.Whf = np.random.randn(h + i, h)
        # Initialize biases for the forward direction
        self.bhf = np.zeros((1, h))
        # Initialize weights for the backward direction
        self.Whb = np.random.randn(h + i, h)
        # Initialize biases for the backward direction
        self.bhb = np.zeros((1, h))
        # Initialize weights for the output
        self.Wy = np.random.randn(2 * h, o)
        # Initialize biases for the output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation for one time
        step in the RNN
        Arguments:
         - h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
            * m is the batch size for the data
         - x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            * m is the batch size for the data
        Returns: H_next, the next hidden state
        """
        # Concatenate the hidden state and the input data
        h_concatenate = np.concatenate((h_prev, x_t), axis=1)
        # Calculate the next hidden state
        h_next = np.tanh(np.dot(h_concatenate, self.Whf) + self.bhf)

        # Return the next hidden state
        return h_next
