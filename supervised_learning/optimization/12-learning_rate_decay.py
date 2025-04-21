#!/usr/bin/env python3
"""
Function that creates a learning rate decay operation in tensorflow
using inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow
    using inverse time decay
    Arguments:
     - alpha is the original learning rate
     - decay_rate is the weight used to determine the rate at which alpha will
        decay
     - decay_step is the number of passes of gradient descent that should occur
        before alpha is decayed further
    Returns:
     The learning rate schedule (callable)
    """
    # Create learning rate decay operation in tf using inverse time decay
    learning_rate_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=decay_step,
        staircase=True
    )

    return learning_rate_decay
