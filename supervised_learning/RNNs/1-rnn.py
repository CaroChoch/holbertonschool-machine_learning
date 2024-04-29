#!/usr/bin/env python3
""" forward propagation for a simple RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Function that performs forward propagation for a simple RNN
    Arguments:
     - rnn_cell is an instance of RNNCell that will be used for the forward
         propagation
     - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
         * t is the maximum number of time steps
         * m is the batch size
         * i is the dimensionality of the data
     - h_0 is the initial hidden state given as a numpy.ndarray of shape (m, h)
         * h is the dimensionality of the hidden state
    Returns: H, Y
     - H is a numpy.ndarray containing all of the hidden states
     - Y is a numpy.ndarray containing all of the outputs
    """
    # Get the shapes of X (data) and h_0 (initial hidden state)
    t, m, i = X.shape
    m, h = h_0.shape
    # Initialize H (hidden state) and Y (output)
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    # Assign the initial hidden state to the first position of H
    H[0] = h_0

    # Iterate over the time steps
    for step in range(t):
        # Get the current input data
        x_t = X[step]
        # Perform one step of the RNN
        h_next, y = rnn_cell.forward(H[step], x_t)
        # Update the hidden state
        H[step + 1] = h_next
        # Update the output
        Y[step] = y
    # Return the hidden states and outputs
    return H, Y
