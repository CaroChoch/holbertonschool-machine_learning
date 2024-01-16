#!/usr/bin/env python3
"""
Function that updates the learning rate using inverse time decay in numpy
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that updates the learning rate using inverse time decay in numpy
    Arguments:
     - alpha is the original learning rate
     - decay_rate is the weight used to determine the rate at which alpha will
        decay
     - global_step is the number of passes of gradient descent that have
        elapsed
     - decay_step is the number of passes of gradient descent that should occur
        before alpha is decayed further
    Returns:
     The updated value for alpha
    """
    # compute epoch number based on the number of passes of gradient descent
    epoch_number = np.floor(global_step / decay_step)
    # learning_rate = alpha / (1 + decay_rate * epoch_number)
    learning_rate = alpha / (1 + decay_rate * epoch_number)
    return learning_rate
