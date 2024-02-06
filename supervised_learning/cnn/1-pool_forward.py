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

    # Calculate dimensions of the output
    h_new = int(h_prev - kh) // sh + 1
    w_new = int(w_prev - kw) // sw + 1

    # Initialize the output
    A_new = np.zeros((m, h_new, w_new, c_prev))

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')

    for i in range(h_new):
        for j in range(w_new):
            # Calculate the starting indices for the convolution operation
            start_h = i * sh
            start_w = j * sw
            # Extract the Region of Interest (ROI)
            roi = A_prev_padded[:, start_h:start_h+kh, start_w:start_w+kw]
            
            # Apply max pooling
        if mode == 'max':
            # For each channel, take the maximum value within the ROI
            A_new[:, i, j, :] = np.max(roi, axis=(1, 2))
        # Apply average pooling
        elif mode == 'avg':
            # For each channel, take the average value within the ROI
            A_new[:, i, j, :] = np.mean(roi, axis=(1, 2))

    return A_new
