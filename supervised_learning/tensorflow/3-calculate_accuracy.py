#!/usr/bin/env python3
""" Function that calculates the accuracy of a prediction """
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction

    Arguments:
     - y is a placeholder for the labels of the input data
     - y_pred is a tensor containing the network’s predictions

    Returns:
        A tensor containing the decimal accuracy of the prediction
        """
    # Compare the prediction with the labels
    prediction = tf.math.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    # Calculate the accuracy of the prediction and convert tensor bool to float
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy
