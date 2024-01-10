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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Updates private attribute __cache
        Returns the output of the neural network and the cache, respectively
        """
        # Initialize A0 as X
        self.__cache["A0"] = X

        # Loop through the layers of the neural network
        for i in range(1, self.__L + 1):
            # z1 = W1.X1 + b1
            # z2 = W2.A1 + b2
            # ...
            # zi = Wi.Ai-1 + bi
            # ...
            # zL = WL.AL-1 + bL
            zi = np.matmul(self.__weights["W" + str(i)], self.__cache[
                "A" + str(i - 1)]) + self.__weights["b" + str(i)]
            # Ai = gzi
            # g represents the activation function
            # tanh activation function
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-zi))

        # Return AL, A0
        # AL is the value of the activation function at the last layer
        # A0 is the value of the activation function at the first layer
        return self.__cache["A" + str(self.L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        Returns the cost
        """
        # Compute the number of examples
        m = Y.shape[1]

        # Compute the cost
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / m

        # Return the cost
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        Returns the neuron’s prediction and the cost of the network,
            respectively
        """
        # Calculate the output of the neural network
        A, An = self.forward_prop(X)

        # Calculate the cost
        cost = self.cost(Y, A)

        # Calculate the predicted labels
        # The predicted labels are 1 if A >= 0.5, 0 otherwise
        predictions = np.where(A >= 0.5, 1, 0)

        # Return the predicted labels and the cost of the network
        return predictions, cost
