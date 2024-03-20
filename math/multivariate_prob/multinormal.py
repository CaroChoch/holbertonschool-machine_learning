#!/usr/bin/env python3
"""
class Multinormal
"""

import numpy as np


class MultiNormal:
    """
    Multinormal class
    """

    def __init__(self, data):
        """
        Initializes a Multinormal distribution with the given data.
        Argument:
         - data is a numpy.ndarray of shape (d, n) containing the data set:
        """
        # Check if data is a 2D numpy.ndarray
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        # Check if data contains multiple data points
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        # Calculate the mean and covariance of the data set
        self.mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (
            data.shape[1] - 1)
