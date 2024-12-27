#!/usr/bin/env python3
""" Function that calculates the softmax cross-entropy loss of a prediction """
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross-entropy loss of a prediction

    Arguments:
     - y is a placeholder for the labels of the input data
     - y_pred is a tensor containing the network’s predictions

    Returns:
        A tensor containing the loss of the prediction
        """
    # Calculate the loss
    loss = tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)

    # Return the loss
    return loss
