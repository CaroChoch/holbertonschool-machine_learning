#!/usr/bin/env python3
""" Likelihood """

import numpy as np


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects
    Arguments:
        * x is the number of patients that develop severe side effects
        * n is the total number of patients observed
        * P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects

    Returns: a 1D numpy.ndarray containing the likelihood of obtaining the
        data, x and n, for each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any([value < 0 or value > 1 for value in P]):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the likelihood of obtaining the data
    # P(x|n, P) = (n!/(x!(n-x)!)) * P^x * (1-P)^(n-x)
    fact = np.math.factorial

    likelihood = fact(n) / (fact(x) * fact(n - x)) * (P ** x) * (
        (1 - P) ** (n - x))

    return likelihood
