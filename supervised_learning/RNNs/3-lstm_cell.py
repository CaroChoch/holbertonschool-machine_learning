#!/usr/bin/env python3
""" Class LSTM Cell that represents an LSTM unit """
import numpy as np


class LSTMCell:
    """ Class that represents an LSTM unit """
    def __init__(self, i, h, o):
        """
        Class constructor for LSTMCell
        Arguments:
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs
        """
        # Initialization of Weights of the cell for the concatenated hidden
        # state and input data (forget gate)
        self.Wf = np.random.randn(h + i, h)
        # Initialization of Weights of the cell for the concatenated hidden
        # state and input data (update gate)
        self.Wu = np.random.randn(h + i, h)
        # Initialization of Weights of the cell for the concatenated hidden
        # state and input data (intermediate cell state)
        self.Wc = np.random.randn(h + i, h)
        # Initialization of Weights of the cell for the concatenated hidden
        # state and input data (output gate)
        self.Wo = np.random.randn(h + i, h)
        # Initialization of Weights of the cell that will be used in the output
        self.Wy = np.random.randn(h, o)
        # Initialization of biases of the cell for the concatenated hidden
        # state and input data (forget gate)
        self.bf = np.zeros((1, h))
        # Initialization of biases of the cell for the concatenated hidden
        # state and input data (update gate)
        self.bu = np.zeros((1, h))
        # Initialization of biases of the cell for the concatenated hidden
        # state and input data (intermediate cell state)
        self.bc = np.zeros((1, h))
        # Initialization of biases of the cell for the concatenated hidden
        # state and input data (output gate)
        self.bo = np.zeros((1, h))
        # Initialization of biases of the cell that will be used in the output
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Function that performs softmax
        Arguments:
            - x is a numpy.ndarray of shape (m, n) containing the input data
                * m is the number of data points
                * n is the number of classes
        Returns:
            A numpy.ndarray of shape (m, n) containing the softmax
            activation for each data point
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """
        Function that performs sigmoid
        Arguments:
            - x is a numpy.ndarray of any shape
        Returns:
            - A numpy.ndarray of the same shape as x containing the"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        Method that performs forward propagation for one time step
        Arguments:
            - h_prev is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
                * m is the batch size for the data
            - c_prev is a numpy.ndarray of shape (m, h) containing the previous
                cell state
            - x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
                * m is the batch size for the data
        Returns: h_next, c_next, y
            - h_next is the next hidden state
            - c_next is the next cell state
            - y is the output of the cell
        """
        # Concatenate the hidden state and the input data
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calculate the forget gate (f)
        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        # Calculate the update gate (u)
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        # Calculate the candidate hidden state (c_hat)
        c_hat = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        # Calculate the cell state (c_next)
        c_next = f * c_prev + u * c_hat
        # Calculate the output gate (o)
        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        # Calculate the next hidden state (h_next)
        h_next = o * np.tanh(c_next)
        # Calculate the output of the cell (y)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        # Return the next hidden state, the next cell state, and the output
        return h_next, c_next, y
