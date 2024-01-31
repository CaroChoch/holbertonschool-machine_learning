#!/usr/bin/env python3
""" Function that performs a same convolution on grayscale images """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images
    Arguments:
        - images is a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
                * m is the number of images
                * h is the height in pixels of the images
                * w is the width in pixels of the images
        - jernel is a numpy.ndarray with shape (jh, jw) containing the jernel
                for the convolution
                * jh is the height of the jernel
                * jw is the width of the jernel
    """
    m, h, w = images.shape  # Get the dimensions of the image
    jh, jw = kernel.shape  # Get the dimensions of the jernel

    # Calculate padding size to ensure same-sized output
    pad_h = jh // 2
    pad_w = jw // 2

    # Pad the input images
    padded_images = np.pad(images,
                           ((0, 0), (pad_h, pad_h),
                            (pad_w, pad_w)), mode='constant')

    # Initialize the output
    output = np.zeros((m, h, w))

    # Perform convolution with 2 loops
    for i in range(h):
        for j in range(w):
            # Extract the Region of Interest (ROI) from the padded image
            roi = padded_images[:, i:i+jh, j:j+jw]

            # Apply convolution by multiplying the ROI with the jernel
            # and summing the results
            output[:, i, j] = np.sum(roi * kernel, axis=(1, 2))

    return output
