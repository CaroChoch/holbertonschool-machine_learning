#!/usr/bin/env python3
"""
Class DeepNeuralNetwork that defines a deep neural network performing
binary classification
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check if layers is not a list or if it is an empty list
        if not isinstance(layers, list) or len(layers) == 0:
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
            if not isinstance(layers[i], int) or layers[i] < 0:
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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        cache is a dictionary containing all the intermediary values of the
            network
        alpha is the learning rate
        Updates the private attribute __weights
        """
        # Compute the number of examples
        m = Y.shape[1]

        # Compute the derivative of the output of the neural network
        dz = cache["A" + str(self.__L)] - Y

        # Loop through the layers of the neural network in reverse order
        for i in range(self.__L, 0, -1):
            # Compute the derivative of the weights at each layer
            dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m

            # Compute the derivative of the biases at each layer
            db = np.sum(dz, axis=1, keepdims=True) / m

            # Compute the derivative of the output of the previous layer
            dz = np.matmul(self.__weights["W" + str(i)].T, dz) * (
                cache["A" + str(i - 1)] * (1 - cache["A" + str(i - 1)]))

            # Update the weights and biases of the network
            self.__weights["W" + str(i)] -= alpha * dw
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        verbose is a boolean that defines whether or not to print information
            about the training
        graph is a boolean that defines whether or not to graph information
            about the training once the training has completed
        Updates the private attributes __weights and __cache
        Returns the evaluation of the training data after iterations of
            training have occurred
        """
        # Check if iterations is an integer
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        # Check if iterations is a positive integer
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        # Check if alpha is a float
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        # Check if alpha is a positive float
        if alpha < 0:
            raise ValueError("alpha must be positive")

        # Check if step is an integer
        if not isinstance(step, int):
            raise TypeError("step must be an integer")

        # Check if verbose or graph is True
        if verbose or graph:
            # Check if step is an integer
            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            # Check if step is positive and less than or equal to iterations
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize list to hold cost of the neural network over iterations
        costs = []

        # Create a list of iterations
        iterations_list = []

        # Loop through the iterations of the neural network
        for i in range(iterations + 1):
            # Calculate the output of the neural network
            self.__A, self.__An = self.forward_prop(X)

            # Perform gradient descent for every iteration except the final one
            if i < iterations:
                self.gradient_descent(Y, self.__An, alpha)

            # Calculate the cost
            cost = self.cost(Y, self.__A)

            # Append the cost to the list of costs
            costs.append(cost)
            # Append the current iteration to the list of iterations
            iterations_list.append(i)

            # Check if the current iteration is a multiple of step or
            # of the first or of the specified final iterations
            if i % step == 0 or i == iterations or i == 0:
                # Print the cost every step iterations if verbose is True
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph is True:
            # Plot the cost over iterations
            plt.plot(np.arange(0, iterations + 1), costs)

            # Label the x-axis
            plt.xlabel("iteration")

            # Label the y-axis
            plt.ylabel("cost")

            # Title the graph
            plt.title("Training Cost")

            # Display the graph
            plt.show()

        # Return the evaluation of the training data
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        filename is the file to which the object should be saved
        If filename does not have the extension .pkl, add it
        """
        # Check if filename does not have the extension .pkl
        if filename[-4:] != ".pkl":
            # Add the extension .pkl to filename
            filename += ".pkl"

        # Save the object to the file
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        filename is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist
        """
        # Check if filename doesn't exist
        if not os.path.exists(filename):
            # Return None
            return None

        # Load the object from the file
        with open(filename, "rb") as f:
            obj = pickle.load(f)

        # Return the loaded object
        return obj
