#!/usr/bin/env python3
""" Perform a random crop of an image """
import tensorflow as tf


def crop_image(image, size):
    """
    performs a random crop of an image
    Arguments:
     - image is a 3D tf.Tensor containing the image to crop
     - size is a tuple containing the size of the crop
    Returns:
     A 3D tf.Tensor containing the cropped image
    """
    # crop the image
    crop = tf.image.random_crop(image, size)

    # return the cropped image
    return crop
