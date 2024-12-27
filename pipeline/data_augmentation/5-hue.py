#!/usr/bin/env python3
""" Changes the hue of an image """
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image
    Arguments:
     - image is a 3D tf.Tensor containing the image to change
     - delta is a float representing the amount the hue should be changed
    Returns:
     The altered image
    """
    # hue adjustment
    hue = tf.image.adjust_hue(image=image, delta=delta)

    # return the adjusted image
    return hue
