#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """ class representing a single neuron """
    def __init__(self, nx):
        """ initializesd the neuron """
        # Check if nx is an integer
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be positive")

        # Initialize weights (W) with a random normal distribution
        self.W = np.random.normal(0, 1, (1, nx))
        # Initialize bias (b) to 0
        self.b = 0
        # Initialize activated output (A) to 0
        self.A = 0
