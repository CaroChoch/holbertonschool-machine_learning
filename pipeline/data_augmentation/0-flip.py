#!/usr/bin/env python3
""" flips an image horizontally """
import tensorflow as tf


def flip_image(image):
    """
    flips an image horizontally
    Arguments:
     - image is a 3D tf.Tensor containing the image to flip
     Returns:
        the flipped image
    """
    # flip the image
    flip = tf.image.flip_left_right(image)

    # return the flipped image
    return flip
