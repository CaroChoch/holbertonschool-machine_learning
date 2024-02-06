#!/usr/bin/env python3
"""
Function that performs forward propagation over a pooling layer of a
neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs forward propagation over a pooling layer of a
neural network
    Arguments:
        - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            * m is the number of examples
            * h_prev is the height of the previous layer
            * w_prev is the width of the previous layer
            * c_prev is the number of channels in the previous layer
        - kernel_shape is a tuple of (kh, kw) containing the size of the
        kernel for the pooling
            * kh is the filter height
            * kw is the filter width
        - stride is a tuple of (sh, sw) containing the strides for the
        pooling
            * sh is the stride for the height
            * sw is the stride for the width
        - mode is a string containing either max or avg, indicating whether
        to perform maximum or average pooling, respectively
    Returns:
        The output of the pooling layer
    """
    # Get dimensions of output layer
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Get dimensions of kernel
    kh, kw = kernel_shape

    # Get stride dimensions
    sh, sw = stride

    # Calculate the dimensions of the output layer
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Initialize the output matrix
    A_out = np.zeros((m, h_out, w_out, c_prev))

    # Loop through the output layer
    for i in range(h_out):
        for j in range(w_out):
            # Calculate the starting and ending indices of the
            # resdpective field
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract the ROI (region of interest) from the input
            roi = A_prev[:, h_start:h_end, w_start:w_end, :]

            # Perform pooling operation based on the specified mode
            if mode == 'max':
                A_out[:, i, j, :] = np.max(roi, axis=(1, 2))
            elif mode == 'avg':
                A_out[:, i, j, :] = np.mean(roi, axis=(1, 2))

    # return the output of the pooling layer
    return A_out
