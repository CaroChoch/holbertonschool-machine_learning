#!/usr/bin/env python3
"""
Function that performs forward propagation over a convolutional layer
of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional layer of
    a neural network
    Arguments:
        - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            * m is the number of examples
            * h_prev is the height of the previous layer
            * w_prev is the width of the previous layer
            * c_prev is the number of channels in the previous layer
        - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
            * kh is the filter height
            * kw is the filter width
            * c_prev is the number of channels in the previous layer
            * c_new is the number of channels in the output
        - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
        - activation is a function that returns the output of the convolutiona
        layer
        - padding is a string that is either same or valid, indicating the type
        of padding used
        - stride is a tuple of (sh, sw) containing the strides for the
        convolution
            * sh is the stride for the height
            * sw is the stride for the width
    Returns:
        The output of the convolutional layer
    """
    # Get dimensions of output layer
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Get dimensions of kernel
    kh, kw, c_prev, c_new = W.shape

    # Get stride dimensions
    sh, sw = stride

    # Calculate padding and output size
    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Calculate dimensions of the output
    h_new = int(h_prev + 2 * ph - kh) // sh + 1
    w_new = int(w_prev + 2 * pw - kw) // sw + 1

    # Initialize the output
    A_new = np.zeros((m, h_new, w_new, c_new))

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Calculate the starting indices for the convolution operation
                start_h = i * sh
                start_w = j * sw
                # Extract the Region of Interest (ROI)
                roi = A_prev_padded[:, start_h:start_h+kh, start_w:start_w+kw]

                A_new[:, i, j, k] = np.sum(
                    roi * W[:, :, :, k],
                    axis=(1, 2, 3)
                    )

    # Add activation function
    A_new = activation(A_new)

    return A_new
