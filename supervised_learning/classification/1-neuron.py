#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """ class representing a single neuron """
    def __init__(self, nx):
        """
        Initializesd the neuron
        Argument:
            nx (int): number of input features to the neuron
        """
        # Check if nx is an integer
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be positive")

        # Initialize weights (W) with a random normal distribution
        self.__W = np.random.normal(size=(1, nx))
        # Initialize bias (b) to 0
        self.__b = 0
        # Initialize activated output (A) to 0
        self.__A = 0

    # getter functions
    @property
    def W(self):
        """
        Getter function for private instance W
        return: private weights
        """
        return self.__W

    @property
    def b(self):
        """
        Getter function for private instance b
        return: private bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter function for private instance A
        return: private activated output
    """
        return self.__A
