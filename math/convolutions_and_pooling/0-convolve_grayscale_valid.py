#!/usr/bin/env python3
""" Function that performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images
    Arguments:
        - images is a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
                * m is the number of images
                * h is the height in pixels of the images
                * w is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
                * kh is the height of the kernel
                * kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    # Get dimensions of images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute dimensions of the output
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize array to store convolution output
    output = np.zeros((m, output_h, output_w))

    # Loop through each position of the output
    for i in range(output_h):
        for j in range(output_w):
            # Perform convolution and sum over axes to get result
            output[:, i, j] = np.sum(images[:, i: i + kh, j: j + kw] * kernel,
                                     axis=(1, 2))

    # Return convolution output
    return output
