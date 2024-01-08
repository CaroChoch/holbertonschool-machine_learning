#!/usr/bin/env python3
"""
Class NeuralNetwork that defines a neural network with one hidden layer
performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    Class representing a neural network with one hidden layer performing
    binary classification
    """
    def __init__(self, nx, nodes):
        """
        Initializesd the neural network object
        Argument:
            nx (int): number of input features to the neuron
            nodes (int): number of nodes found in the hidden layer
        """
        # Check if nx is an integer
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check if nodes is an integer
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")

        # Check if nodes is a positive integer
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights vector of the hidden layer (W1) with
        # a random normal distribution
        self.__W1 = np.random.normal(size=(nodes, nx))
        # Initialize bias of the hidden layer (b1) to 0
        self.__b1 = np.zeros((nodes, 1))
        # Initialize activated output of the hidden layer (A1) to 0
        self.__A1 = 0
        # Initialize weights vector of the output neuron (W2) with
        # a random normal distribution
        self.__W2 = np.random.normal(size=(1, nodes))
        # Initialize bias of the output neuron (b2) to 0
        self.__b2 = 0
        # Initialize activated output of the output neuron(prediction)(A2)to 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter function for private instance W1
        return: private weights
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter function for private instance b1
        return: private bias
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter function for private instance A1
        return: private activated output
    """
        return self.__A1

    @property
    def W2(self):
        """
        Getter function for private instance W2
        return: private weights
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter function for private instance b2
        return: private bias
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter function for private instance A2
        return: private activated output
    """
        return self.__A2
