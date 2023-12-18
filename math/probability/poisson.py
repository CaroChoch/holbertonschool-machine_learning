#!/usr/bin/env python3
""" Module containing a class Poisson that represents a poisson distribution"""


class Poisson:
    """
    class representing a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        Arguments:
            data: list of the data to be used to estimate the distribution
            lambtha: the expected number of occurences in a given time frame
        """

        # Check if data is not provided (None)
        if data is None:
            # Check if lambtha is not a positive value
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                # If lambtha is valid, assign it to the instance attribute
                self.lambtha = float(lambtha)
        else:
            # If data is provided
            # Check if data is a list
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # Check if data contains at least two values
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # If conditions are satisfied, calculate lambtha from data
            else:
                self.lambtha = float(sum(data) / len(data))
