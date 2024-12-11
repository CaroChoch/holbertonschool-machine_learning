#!/usr/bin/env python3
""" Rotates the image by 90 degrees counter-clockwise """
import tensorflow as tf


def rotate_image(image):
    """
    Rotates the image by 90 degrees counter-clockwise
    Arguments:
     - image is a 3D tf.Tensor containing the image to rotate
    Returns:
     The rotated image
    """
    # rotate the image
    rotate = tf.image.rot90(image)

    # return the rotated image
    return rotate
