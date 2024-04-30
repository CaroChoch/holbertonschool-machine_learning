#!/usr/bin/env python3
""" Bidirectional RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    Arguments:
        - bi_cell is an instance of BidirectionalCell that will be used for the
            forward propagation
        - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            * t is the maximum number of time steps
            * m is the batch size
            * i is the dimensionality of the data
        - h_0 is the initial hidden state in the forward direction, given as a
            numpy.ndarray of shape (m, h)
            h is the dimensionality of the hidden state
        - h_T is the initial hidden state in the backward direction, given as a
            numpy.ndarray of shape (m, h)
    Returns: H, Y
        - H is a numpy.ndarray containing all of the concatenated hidden
            states
        - Y is a numpy.ndarray containing all of the outputs
    """
    # Get the dimensions of the data
    t, m, i = X.shape
    # Get the dimensions of the hidden states
    h = h_0.shape[1]
    # Initialize hidden states of the forward (Hf) and backward directions (Hb)
    Hf = np.zeros((t + 1, m, h))  # Include the initial hidden state
    Hb = np.zeros((t + 1, m, h))  # Include the initial hidden state
    # Initialize the outputs
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    # Initialize the first hidden states
    Hf[0] = h_0
    # Initialize the last hidden states
    Hb[t] = h_t

    # Perform forward propagation
    for i_step in range(t):
        # Calculate the hidden states in the forward direction
        Hf[i_step + 1] = bi_cell.forward(Hf[i_step], X[i_step])
    for i_step in reversed(range(t)):
        # Calculate the hidden states in the backward direction
        Hb[i_step] = bi_cell.backward(Hb[i_step + 1], X[i_step])
    # Concatenate the hidden states
    H = np.concatenate((Hf[1:], Hb[:t]), axis=-1)
    # Calculate the outputs
    Y = bi_cell.output(H)

    # Return the hidden states and the outputs
    return H, Y
