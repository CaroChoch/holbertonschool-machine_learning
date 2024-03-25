#!/usr/bin/env python3
""" Continuous Posterior """

from scipy import special


def posterior(x, n, p1, p2):
    """
    Function that calculates the posterior probability for the various
    hypothetical probabilities of developing severe side effects
    Arguments:
        * x is the number of patients that develop severe side effects
        * n is the total number of patients observed
        * p1 is the lower bound on the range
        * p2 is the upper bound on the range

    Returns: the posterior probability that p is within the range
        [p1, p2] given x and n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p1 >= p2:
        raise ValueError("p2 must be greater than p1")

    # Calculating the alpha and beta parameters of the beta distribution
    alpha = x + 1
    beta = n - x + 1

    # Calculating the probabilities associated with the lower and upper bounds
    lower_bound_probability = special.betainc(alpha, beta, p1)
    upper_bound_probability = special.betainc(alpha, beta, p2)

    # Calculating the posterior probability
    posterior_probability = upper_bound_probability - lower_bound_probability

    return posterior_probability
