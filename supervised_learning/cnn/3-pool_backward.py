#!/usr/bin/env python3
"""
Function that performs back propagation over a pooling layer of a
neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs back propagation over a pooling layer of
    a neural network
    Arguments:
        - dA is numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the output of the pooling layer
            * m is the number of examples
            * h_new is the height of the output
            * w_new is the width of the output
            * c_new is the number of channels in the output
        - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            * h_prev is the height of the previous layer
            * w_prev is the width of the previous layer
            * c_prev is the number of channels in the previous layer
        - kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
            * kh is the kernel height
            * kw is the kernel width
        - stride is a tuple of (sh, sw) containing the strides for the
        convolution
            * sh is the stride for the height
            * sw is the stride for the width
        - mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively
    Returns:
        The partial derivatives with respect to the previous layer (dA_prev)
    """
    # Extract dimensions from the input arrays
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Get dimensions of the kernel
    kh, kw = kernel_shape

    # Get stride dimensions
    sh, sw = stride

    # Initialize shape for dA_prev
    dA_prev = np.zeros(shape=A_prev.shape)

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
                for ch in range(c_new):
                    # Perform pooling operation based on the specified mode
                    if mode == 'max':
                        A_slice = A_prev[i, h_start:h_end, w_start:w_end, ch]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[i, h_start:h_end,
                                w_start:w_end, ch] += mask * dA[i,
                                                                h, w, ch]
                    elif mode == 'avg':
                        mask = dA[i, h, w, ch] / (kh * kw)
                        dA_prev[i, h_start:h_end,
                                w_start:w_end, ch] += np.ones(
                            shape=(kh, kw)) * mask

    return dA_prev
