#!/usr/bin/env python3
""" Randomly adjusts the contrast of an image """
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image
    Arguments:
     - image is a 3D tf.Tensor representing the input image to adjust the contrast
     - lower is a float representing the lower bound of the random contrast factor range
     - upper is a float representing the upper bound of the random contrast factor range
    Returns:
     The contrast-adjusted image
    """
    # contrast adjustment
    contrast = tf.image.random_contrast(image=image, lower=lower, upper=upper)

    # return the adjusted image
    return contrast
