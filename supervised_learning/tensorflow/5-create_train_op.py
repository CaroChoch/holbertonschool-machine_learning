#!/usr/bin/env python3
""" Function that creates the training operation for the network """
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network

    Arguments:
     - loss is the loss of the network’s prediction
     - alpha is the learning rate

    Returns:
     The operation that trains the network using gradient descent
    """
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=alpha, name="GradientDescent")
    train = optimizer.minimize(loss)
    return train
