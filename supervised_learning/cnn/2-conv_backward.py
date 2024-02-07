#!/usr/bin/env python3
"""
Function that performs back propagation over a convolutional layer of a
neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Function that performs back propagation over a convolutional layer of
    a neural network
    Arguments:
        - dZ is numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the unactivated output of
        the convolutional layer
            * m is the number of examples
            * h_new is the height of the output
            * w_new is the width of the output
            * c_new is the number of channels in the output
        - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            * h_prev is the height of the previous layer
            * w_prev is the width of the previous layer
            * c_prev is the number of channels in the previous layer
        - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
            * kh is the filter height
            * kw is the filter width
        - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
        - padding is a string that is either same or valid, indicating the type
        of padding used
        - stride is a tuple of (sh, sw) containing the strides for the
        convolution
            * sh is the stride for the height
            * sw is the stride for the width
    Returns:
        The partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    # Extract dimensions from the input arrays
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Get dimensions of the kernel
    kh, kw, _, _ = W.shape

    # Get stride dimensions
    sh, sw = stride

    # Calculate padding based on the specified type
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

    elif padding == "valid":
        ph, pw = 0, 0

    # Pad A_prev if required
    A_prev_padded = np.pad(
            A_prev,
            ((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode='constant')

    # Initialize gradients for the previous layer, kernels, and biases
    dA_prev = np.zeros(shape=A_prev_padded.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over examples
    for i in range(m):
        # Loop over height
        for h in range(h_new):
            h_start = h * sh
            h_end = h * sh + kh
            # Loop over width
            for w in range(w_new):
                w_start = w * sw
                w_end = w * sw + kw
                # Loop over channels
                for c in range(c_new):
                    # Compute gradients
                    dA_prev[i, h_start:h_end, w_start:w_end, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c])
                    dW[:, :, :, c] += (
                        A_prev_padded[i, h_start:h_end, w_start:w_end, :] *
                        dZ[i, h, w, c])

    # Maintain output size when padding is "same"
    if padding == "same":
        dA = dA_prev[:, ph:-ph, pw:-pw, :]
    elif padding == "valid":
        dA = dA_prev

    return dA, dW, db
