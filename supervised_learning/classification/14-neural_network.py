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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Argument:
            X (numpy.ndarray): contains the input data
        Return:
            Returns the private attributes __A1 and __A2, respectively
        """
        # Compute the dot product of X and W1, incorporating the bias
        # into each node of the hidden layer
        z = np.matmul(self.__W1, X) + self.__b1
        # Apply sigmoid activation function
        self.__A1 = 1 / (1 + np.exp(-z))

        # Compute the dot product of A1 and W2, incorporating the bias
        # into the output neuron
        z = np.matmul(self.__W2, self.__A1) + self.__b2
        # Apply sigmoid activation function
        self.__A2 = 1 / (1 + np.exp(-z))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
            Y (numpy.ndarray): contains the correct labels for the input data
            A (numpy.ndarray): containing the activated output of the neuron
                               for each example
        Return:
            Returns the cost
        """
        # Cost function
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Arguments:
            X: a numpy.ndarray with shape (nx, m) that contains the input data
        nx: is the number of input features to the neuron
        m: is the number of examples
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        Return:
            Returns the neuron’s prediction and the cost of the network,
        respectively
        """
        # Generate predictions
        A1, A2 = self.forward_prop(X)
        predictions = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Arguments:
            X (numpy.ndarray): contains the input data
            Y (numpy.ndarray): contains the correct labels for the input data
            A1 (numpy.ndarray): containing the activated output of the hidden
                                 layer for each example
            A2 (numpy.ndarray): containing the predicted output of the neuron
                                 for each example
            alpha (float): is the learning rate
        """
        # Calculate number of examples
        m = Y.shape[1]

        # Calculation of derivatives of the output layer
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # Calculation of derivatives of the hidden layer
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # Update of weights and biases
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        Arguments:
            X (numpy.ndarray): contains the input data
            Y (numpy.ndarray): contains the correct labels for the input data
            iterations (int): is the number of iterations to train over
            alpha (float): is the learning rate
        """
        # Check if iterations is an integer
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        # Check if iterations is a positive integer
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        # Check if alpha is a float
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")

        # Check if alpha is a positive float
        if alpha < 0:
            raise ValueError("alpha must be positive")

        # Train the model
        for i in range(iterations):
            # Generate predictions and cost
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
