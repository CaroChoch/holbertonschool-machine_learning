#!/usr/bin/env python3
""" Randomly changes the brightness of an image """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image
    Arguments:
     - image is a 3D tf.Tensor containing the image to change
     - max_delta is a float representing the maximum amount the image should be brightened (or darkened)
    Returns:
     The altered image
    """
    # brightness adjustment
    brightness = tf.image.random_brightness(image=image, max_delta=max_delta)

    # return the adjusted image
    return brightness
