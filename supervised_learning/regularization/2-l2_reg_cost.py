#!/usr/bin/env python3
"""
Function that calculates the cost of a neural network with L2 regularization
"""
import numpy as np
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Function that calculates the cost of a neural network with L2
    regularization
    Arguments:
     - cost is the cost of the network without L2 regularization
     Returns:
         a tensor containing the cost of the network accounting for
         L2 regularization
    """
    # Calculate the regularization cost by adding L2 regularization
    # losses to the base cost
    l2_regularized_cost = cost + tf.losses.get_regularization_losses()
    # Return the final L2 regularized cost
    return l2_regularized_cost
