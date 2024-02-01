#!/usr/bin/env python3
"""
Function that performs a convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on grayscale images
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
        - padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            * if ‘same’, performs a same convolution
            * if ‘valid’, performs a valid convolution
            * if a tuple:
                * ph is the padding for the height of the image
                * pw is the padding for the width of the image
                * the image should be padded with 0’s
        - stride is a tuple of (sh, sw)
            * sh is the stride for the height of the image
            * sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape  # Get the dimensions of the image
    kh, kw = kernel.shape  # Get the dimensions of the kernel
    sh, sw = stride  # Get the dimensions of the stride

    # Calculate padding size to ensure same-sized output
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad the input images
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph),
                            (pw, pw)), mode='constant')

    # Calculate the dimensions of the output
    pad_h = (h + 2 * ph - kh) // sh + 1
    pad_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output
    output = np.zeros((m, pad_h, pad_w))

    # Perform convolution with 2 loops
    for i in range(pad_h):
        for j in range(pad_w):
            # Extract the Region of Interest (ROI) from the padded image
            roi = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]

            # Apply convolution by multiplying the ROI with the kernel
            # and summing the results
            output[:, i, j] = np.sum(roi * kernel, axis=(1, 2))

    return output
