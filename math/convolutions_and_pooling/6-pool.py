#!/usr/bin/env python3
"""
Function that performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    Arguments:
        - images is a numpy.ndarray with shape (m, h, w, c) containing
        multiple images
            * m is the number of images
            * h is the height in pixels of the images
            * w is the width in pixels of the images
            * c is the number of channels in the image
        - kernel_shape is a tuple of (kh, kw) containing the kernel shape for
        the pooling
            * kh is the height of the kernel
            * kw is the width of the kernel
        - stride is a tuple of (sh, sw)
            * sh is the stride for the height of the image
            * sw is the stride for the width of the image
        - mode indicates the type of pooling
            * max indicates max pooling
            * avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """
    # Get dimensions of images, kernel and stride
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate dimensions of the output
    pad_h = (h - kh) // sh + 1
    pad_w = (w - kw) // sw + 1

    # Initialize the output
    output = np.zeros((m, pad_h, pad_w, c))

    # Perform pooling with 2 loops
    for i in range(pad_h):
        for j in range(pad_w):
            # Select the region of interest (ROI) based on the current position
            roi = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]

            # Apply max pooling
            if mode == 'max':
                # For each channel, take the maximum value within the ROI
                output[:, i, j, :] = np.max(roi, axis=(1, 2))
            # Apply average pooling
            elif mode == 'avg':
                # For each channel, take the average value within the ROI
                output[:, i, j, :] = np.mean(roi, axis=(1, 2))

    # Return the pooled images
    return output
