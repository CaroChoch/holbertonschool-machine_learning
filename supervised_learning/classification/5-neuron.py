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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Arguments:
            X: a numpy.ndarray with shape (nx, m) that contains the input data
        nx: is the number of input features to the neuron
        m: is the number of examples
        Return:
            Returns the private attribute __A
        """
        # z = W.X + b
        z = np.matmul(self.__W, X) + self.__b
        sigmoid_function = 1 / (1 + np.exp(-z))
        self.__A = sigmoid_function
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
            A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        Return:
            Returns the cost
        """
        # Cost Function
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
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Arguments:
            X: a numpy.ndarray with shape (nx, m) that contains the input data
        nx: is the number of input features to the neuron
        m: is the number of examples
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
            A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
            alpha: is the learning rate
        Return:
            Updates the private attributes __W and __b
        """
        # Calculate the derivative of the cost function
        m = Y.shape[1]
        # dz represents the difference between the activated output and
        # the correct labels
        dz = A - Y
        # dw represents the gradient of the loss with respect to w
        # dw is the partial derivative of the cost with respect to the
        # weights. It is calculated by multiplying the transpose of X by dz and
        # normalizing by the number of examples m
        dw = np.matmul(X, dz.T) / m
        # db represents the gradient of the loss with respect to b
        # db is the partial derivative of the cost with respect to the bias.
        # It is calculated by summing dz over all the examples and normalizing
        db = np.sum(dz) / m

        # Update the weights and bias
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)
