#!/usr/bin/env python3
""" Marginal Probability """

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


def intersection(x, n, P, Pr):
    """
    Function that calculates the intersection of obtaining this data with the
    various hypothetical probabilities
    Arguments:
        * x is the number of patients that develop severe side effects
        * n is the total number of patients observed
        * P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
        * Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns: a 1D numpy.ndarray containing the intersection of obtaining x and
        n with each probability in P, respectively
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
    if not isinstance(Pr, np.ndarray) or not np.array_equal(Pr.shape, P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any([value < 0 or value > 1 for value in P]):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any([value < 0 or value > 1 for value in Pr]):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the intersection of obtaining the data with the various
    # hypothetical probabilities
    # P(x|n, P) * Pr(P)
    intersection = likelihood(x, n, P) * Pr

    return intersection


def marginal(x, n, P, Pr):
    """
    Function that calculates the marginal probability of obtaining the data
    Arguments:
        * x is the number of patients that develop severe side effects
        * n is the total number of patients observed
        * P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
        * Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns: the marginal probability of obtaining x and n
    """
    # Calculate the marginal probability of obtaining the data
    # P(x|n, P) * Pr(P) + ...
    marginal = np.sum(intersection(x, n, P, Pr))

    return marginal
