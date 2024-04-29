#!/usr/bin/env python3
""" Performs forward propagation for a deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    Arguments:
        - rnn_cells is a list of RNNCell instances of length l that will be
            used for the forward propagation
            * l is the number of layers
        - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            * t is the maximum number of time steps
            * m is the batch size
            * i is the dimensionality of the data
        - h_0 is the initial hidden state, given as a numpy.ndarray of
            shape (l, m, h)
            * h is the dimensionality of the hidden state
    Returns: H, Y
        - H is a numpy.ndarray containing all of the hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """
    # Get the dimensions of the data
    t, m, i = X.shape
    # Get the dimensions of the initial hidden state
    l, m, h = h_0.shape
    # Initialize the hidden states (H) and the outputs (Y)
    H = np.zeros((t + 1, l, m, h))
    Y = []
    # Initialize the first hidden state
    H[0] = h_0

    # Perform forward propagation
    for i in range(t):
        for j in range(l):
            # Get the current hidden state
            if j == 0:
                # If it is the first layer, use the input data
                h, y = rnn_cells[j].forward(H[i][j], X[i])
            else:
                # Otherwise, use the hidden states from the previous layer
                h, y = rnn_cells[j].forward(H[i][j], h)
            H[i + 1, j] = h
        Y.append(y)
    Y = np.array(Y)
    # Return the hidden states and the outputs
    return H, Y
