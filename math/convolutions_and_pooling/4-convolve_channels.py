#!/usr/bin/env python3
"""
Function that performs a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels
    Arguments:
        - images is a numpy.ndarray with shape (m, h, w, c) containing
        multiple images
            * m is the number of images
            * h is the height in pixels of the images
            * w is the width in pixels of the images
            * c is the number of channels in the image
        - kernel is a numpy.ndarray with shape (kh, kw, c) containing the
        kernel for the convolution
            * kh is the height of the kernel
            * kw is the width of the kernel
            * c is the number of channels in the image
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
    m, h, w, c = images.shape  # Get dimensions of images
    kh, kw, c = kernel.shape    # Get dimensions of kernel
    sh, sw = stride            # Get stride dimensions

    # Calculate padding size to ensure same-sized output
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Calculate dimensions of the output
    pad_h = (h + 2 * ph - kh) // sh + 1
    pad_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output
    output = np.zeros((m, pad_h, pad_w))

    # Pad the input images
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph),
                            (pw, pw), (0, 0)), mode='constant')

    # Perform convolution with 2 loops
    for i in range(pad_h):
        for j in range(pad_w):
            # Extract the Region of Interest (ROI) from the padded image
            roi = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]

            # Apply convolution by multiplying the ROI with the kernel
            # and summing the results
            output[:, i, j] = np.sum(roi * kernel, axis=(1, 2, 3))

    return output
