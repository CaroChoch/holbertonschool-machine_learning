#!/usr/bin/env python3
""" Function that returns two placeholders x and y for the neural network """
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders x and y for the neural network

    Arguments:
     - nx is the number of feature columns in our data
     - classes is the number of classes in our classifier
     """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y
