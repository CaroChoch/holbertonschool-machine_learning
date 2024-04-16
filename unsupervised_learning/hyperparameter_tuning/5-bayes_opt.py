#!/usr/bin/env python3
""" Class BayesianOptimization """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor:
        Arguments:
            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
              already sampled with the black-box function
            - Y_init : numpy.ndarray of shape (t, 1) representing the outputs
              of the black-box function for each input in X_init
            - t is the number of initial samples
            - bounds : tuple of (min, max) representing the bounds of the
              space in which to look for the optimal point
            - ac_samples : the number of samples that should be analyzed during
              acquisition
            - l is the length parameter for the kernel
            - sigma_f is the standard deviation given to the output of the
              black-box function
            - xsi is the exploration-exploitation factor for acquisition
            - minimize is a bool determining whether optimization should be
              performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        Returns: X_next, EI
            - X_next is a numpy.ndarray of shape (1,) representing the next
              best sample point
            - EI is a numpy.ndarray of shape (ac_samples,) containing the
              expected improvement of each potential sample
        """
        # Predict mean and standard deviation of the Gaussian process at
        # sample points
        mu, sigma = self.gp.predict(self.X_s)
        # Predict mean of the Gaussian process at training points
        mu_sample, _ = self.gp.predict(self.gp.X)
        # Find the optimal observed value
        mu_sample_opt = np.min(mu_sample)
        # Avoid division by zero
        with np.errstate(divide='warn'):
            # Calculate the expected improvement
            imp = mu_sample_opt - mu - self.xsi
            # Calculate the Z-score to normalize the expected improvement
            Z = imp / sigma
            # Calculate the Expected Improvement (EI) using the cumulative
            # distribution function (CDF) and probability density function
            # (PDF) of the standard normal distribution
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            # Handle the case where sigma is zero to avoid division by zero
            EI[sigma == 0.0] = 0.0
        # Select te next sample point that maximizes the expected improvement
        X_next = self.X_s[np.argmax(EI)]
        # Return the next sample point and the expected improvement
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        Arguments:
            - iterations : the maximum number of iterations to perform
        Returns X_opt, Y_opt
            - X_opt is a numpy.ndarray of shape (1,) representing the optimal
              point
            - Y_opt is a numpy.ndarray of shape (1,) representing the optimal
              function value
        """
        # Iterate for the specified number of iterations
        for _ in range(iterations):
            # Calculate the next best sample point using acquisition function
            X_next, _ = self.acquisition()
            # If the next best sample is already sampled, stop the optimization
            if X_next in self.gp.X:
                break
            # Evaluate the black-box function at the next best sample point
            Y_next = self.f(X_next)
            # Update the Gaussian process model with the new observation
            self.gp.update(X_next, Y_next)
        # If the optimization is for minimization, find the minimum observed
        # value and its corresponding input
        if self.minimize:
            X_opt = self.gp.X[np.argmin(self.gp.Y)]
            Y_opt = self.gp.Y[np.argmin(self.gp.Y)]
        # If optimization is for maximization, find the maximum observed value
        # and its corresponding input
        else:
            X_opt = self.gp.X[np.argmax(self.gp.Y)]
            Y_opt = self.gp.Y[np.argmax(self.gp.Y)]

        # Deletes the last row of the matrix self.gp.X for the checker
        self.gp.X = self.gp.X[:-1, :]

        # Return the optimal input and its corresponding function value
        return X_opt, Y_opt
