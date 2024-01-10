#!/usr/bin/env python3
"""
Class DeepNeuralNetwork that defines a deep neural network performing
binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork that defines a deep neural network performing
    binary classification
    """
    def __init__(self, nx, layers):
        """
        Initializesd the deep neural network object
        Argument:
            nx (int): number of input features to the deep neural network
            layers (int): number of layers in the deep neural network
        """
        # Check if nx is an integer
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check if layers is not a list or if it is an empty list
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Initialize L
        # L represents the number of layers in the neural network
        self.__L = len(layers)

        # Initialize cache
        # cache holds all intermediary values of the network
        # cache is initialized to an empty dictionary
        self.__cache = {}

        # Initialize weights and biases of the network
        # based on He et al. method and activated outputs of each layer
        # Initialize empty dictionary to hold weights and biases
        self.__weights = {}

        # Loop through the layers of the neural network
        for i in range(self.L):
            # Check if the type of the current element in layers is not an
            # integer or if it is a negative integer
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            # Check if it's the first layer
            if i == 0:
                # Initialize weights of the first layer using He et al. method
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                # Initialize biases of the first layer to zeros
                self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

            else:
                # Initialize weights of hidden layers using He et al. method
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                # Initialize biases of hidden layers to zeros
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Getter function for L
        Returns L
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter function for cache
        Returns cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter function for weights
        Returns weights
        """
        return self.__weights
