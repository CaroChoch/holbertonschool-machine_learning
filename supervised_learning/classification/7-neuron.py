#!/usr/bin/env python3
"""
module containing class Neuron:
Class constructor def __init__(self, nx)
Private instance attributes: __W, __b, __A
Public method def forward_prop(self, X)
Public method def cost(self, Y, A)
Public method def evaluate(self, X, Y)
Public method def gradient_descent(self, X, Y, A, alpha=0.05)
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Class Neuron : defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Class contructor
        Args:
            nx: the number of input features to the neuron
        Private instance attributes:
            W: weights vector for the neuron
            b: bias for the neuron
            A: activated output of the neuron (prediction)
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function
        __W: weights vector for the neuron
        return private instance attribute __W
        """
        return self.__W

    @property
    def b(self):
        """
        getter function
        __b: bias for the neuron
        return private instance attribute __b
        """
        return self.__b

    @property
    def A(self):
        """
        getter function
        __A: activated output of the neuron (prediction)
        return private instance attribute __A
        """
        return self.__A

    def forward_prop(self, X):
        """
        Public method that calculates the forward propagation of the neuron
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
            with nx: number of input features to the neuron
            and m: number of examples
        Returns the updated private attribute __A with
        sigmoid activation function :
            1 / 1 + e^(-x) with x = Sum(Wi*Xi + b) for i=0 to nx
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Public method that calculates the cost of the model
        using logistic regression
        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data with m: number of examples
            A: numpy.ndarray with shape (1, m) containing the activated output
                of the neuron for each example
        Returns the cost:
            cost = -1/m * Sum(Yi*log(Ai) + (1-Yi)*log(1-Ai))
        """
        m = Y.shape[1]
        cost = - (1 / m) * \
            (np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Public method that evaluates the neuron's predictions
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                with nx: number of input features to the neuron
                and m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
        Returns the neuron's prediction: numpy.ndarray with shape (1, m)
            containing the predicted labels for each example
            and the cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Public method that calculates one pass of gradient descent
        on the neuron
        Updates the private attributes __W and __b
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                with nx: number of input features to the neuron
                and m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
                of the neuron for each example
            alpha: the learning rate
        """
        m = X.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Public method that trains the neuron
        Updates the private attributes __W, __b and __A
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                with nx: number of input features to the neuron
                and m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            iterations: the number of iterations to train over
            alpha: the learning rate
            verbose: a boolean that defines whether or not to print information
                about the training
            graph: a boolean that defines whether or not to graph information
                about the training once the training has completed.
            step: the step of the training
        """
        # Check if iterations is an integer
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")

        # Check if iterations is a positive integer
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        # Check if alpha is a float
        if type(alpha) != float:
            raise TypeError("alpha must be a float")

        # Check if alpha is positive
        if alpha < 0:
            raise ValueError("alpha must be positive")

        # Check if verbose or graph is True
        if verbose or graph:
            # Check if step is an integer
            if type(step) != int:
                raise TypeError("step must be an integer")

            # Check if step is positive and less than or equal to iterations
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        # Create a list of costs
        costs = []

        # Create a list of iterations
        iterations_list = []

        # Iterate through the iterations
        for i in range(iterations + 1):
            # Call the forward propagation function
            self.__A = self.forward_prop(X)

            # Call the gradient descent function
            self.gradient_descent(X, Y, self.__A, alpha)

            # Append the cost to the list of costs
            cost = self.cost(Y, self.__A)

            # Check if the current iteration is a multiple of step or
            # of the first or of the specified final iterations
            if i % step == 0 or i == iterations or i == 0:
                costs.append(cost)
                iterations_list.append(i)

                # Print the cost every step iterations if verbose is True
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))

        # Plot the training data every step iterations if graph is True
        if graph is True:
            plt.plot(iterations_list, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Return evaluation of the training data after iterations of training
        return self.evaluate(X, Y)
