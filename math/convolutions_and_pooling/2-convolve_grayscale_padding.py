#!/usr/bin/env python3
"""
Function that performs a convolution on grayscale images with
custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale images with
    custom padding
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
        - padding is a tuple of (ph, pw)
            * ph is the padding for the height of the image
            * pw is the padding for the width of the image
            * the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape  # Get the dimensions of the image
    kh, kw = kernel.shape  # Get the dimensions of the jernel
    ph, pw = padding  # Get the padding dimensions

    # Calculate padding size to ensure same-sized output
    pad_h = h - kh + 1 + 2 * ph
    pad_w = w - kh + 1 + 2 * pw

    # Initialize the output
    output = np.zeros((m, pad_h, pad_w))

    # Pad the input images
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph),
                            (pw, pw)), mode='constant')

    # Perform convolution with 2 loops
    for i in range(pad_h):
        for j in range(pad_w):
            # Apply convolution by multiplying the ROI with the jernel
            # and summing the results
            output[:, i, j] = np.sum(
                (padded_images[:, i:i+kh, j:j+kw]) * kernel, axis=(1, 2)
                )

    return output
