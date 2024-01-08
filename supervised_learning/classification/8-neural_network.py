#!/usr/bin/env python3
"""
Class Neuron that defines a neural network with one hidden layer performing
binary classification
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

        # Initialize weights vector for the hidden layer (W1) with
        # a random normal distribution
        self.W1 = np.random.normal(size=(nodes, nx))
        # Initialize bias for the hidden layer (b1) to 0
        self.b1 = np.zeros((nodes, 1))
        # Initialize activated output for the hidden layer (A1) to 0
        self.A1 = 0
        # Initialize weights vector for the output neuron (W2) with
        # a random normal distribution
        self.W2 = np.random.normal(size=(1, nodes))
        # Initialize bias for the output neuron (b2) to 0
        self.b2 = 0
        # Initialize activated output for the output neuron(prediction)(A2)to 0
        self.A2 = 0
