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

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        Argument:
         - x is a numpy.ndarray of shape (d, 1) containing the data point
        Returns: the value of the PDF
        """
        # Check if x is a 2D numpy.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        # Get the number of dimensions
        d = self.mean.shape[0]

        # Check if x has the same number of dimensions as the mean
        if x.shape[0] != self.mean.shape[0] or x.shape[1] != 1:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        # Calculate the difference between the data point and the mean
        x_m = x - self.mean

        # Calculate the covariance determinant
        cov_det = np.linalg.det(self.cov)

        # Calculate the exponent of the PDF
        exponent = -0.5 * np.dot(np.dot(x_m.T, np.linalg.inv(self.cov)), x_m)

        # Calculate the denominator part of the PDF formula
        denominator_part = np.sqrt((2 * np.pi) ** d * cov_det)

        # Calculate the PDF of the data point
        pdf = np.exp(exponent) / denominator_part

        return pdf.item()  # Convert the result to a Python float
