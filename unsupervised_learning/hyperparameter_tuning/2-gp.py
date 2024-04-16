#!/usr/bin/env python3
""" Class Gaussian Process """
import numpy as np


class GaussianProcess:
    """
    Class Gaussian Process that represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        Arguments:
            - X_init : numpy.ndarray of shape (t, 1) representing the inputs
            - Y_init : numpy.ndarray of shape (t, 1) representing the outputs
            - t : number of initial samples
            - l : the length parameter for the kernel
            - sigma_f : standard deviation given to the output of the
            black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Public instance method that calculates the covariance kernel
        matrix between two matrices
        Arguments:
        - X1 is a numpy.ndarray of shape (m, 1)
        - X2 is a numpy.ndarray of shape (n, 1)
        Returns:
        The covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """

        # Calculating squared Euclidean distances between all pairs of rows
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) \
            + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

        # Applying the covariance kernel function
        # The kernel used here is the squared exponential kernel
        # k(x1, x2) = sigma_f^2 * exp(-0.5 / l^2 * ||x1 - x2||^2)
        # where sigma_f^2 is the signal variance, l is the lengthscale,
        # and ||x1 - x2||^2 is the squared Euclidean distance
        covariance_kernel = self.sigma_f ** 2 * np.exp(
            -0.5 / self.l ** 2 * sqdist)

        return covariance_kernel

    def predict(self, X_s):
        """
        Public instance method that predicts the mean and standard deviation
        of points in a Gaussian process
        Arguments:
            - X_s : numpy.ndarray of shape (s, 1) containing all of the points
            whose mean and standard deviation should be calculated
        Returns: mu, sigma
            - mu : numpy.ndarray of shape (s,) containing the mean for each
        point in X_s, respectively
            - sigma : numpy.ndarray of shape (s,) containing the variance for
        each point in X_s, respectively
        """
        # Calculating the kernel matrix between training points and test points
        K_s = self.kernel(self.X, X_s)
        # Calculatring the kernel matrix among test points
        K_ss = self.kernel(X_s, X_s)
        # Inverting the kernel matrix of training points
        K_inv = np.linalg.inv(self.K)

        # Calculating the mean prediction for test points
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # Calculating the variance prediction (sigma) fot test points
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Public instance method that updates a Gaussian Process
        Arguments:
            - X_new : numpy.ndarray of shape (1,) that represents the new
                sample point
            - Y_new : numpy.ndarray of shape (1,) that represents the new
                sample function value
        """
        # Updating the Gaussian Process
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
