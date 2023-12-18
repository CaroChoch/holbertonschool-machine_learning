#!/usr/bin/env python3
""" Module containing a class Exponential that represents an
exponential distribution """


class Exponential:
    """
    Class that represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        class constructor function
        Arguments:
            data: list of the data to be used to estimate the distribition
            lambtha: the expected number of occurences in a given time frame
        """

        if data is not None:
            # If data is given, calculate lambtha from the data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            # If data does not contain at least two data points,
            # raise a ValueError
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate lambtha as the inverse of the mean of the data
            self.lambtha = 1. / (sum(data) / len(data))

        else:
            # If data is not given, use the provided lambtha
            self.lambtha = float(lambtha)

        # Check if lambtha is a positive value
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

    def pdf(self, x):
        """
        Instance method that calculates the value of the PDF for
        a given time period
        Argument:
            x: the time period
        Return: the PDF value for x, Otherwise 0 if x is out of range
        """

        e_value = 2.7182818285

        # If x is out of range, return 0
        if x < 0:
            return 0
        # Calculates the value of the PDF
        pdf = self.lambtha * e_value ** (-self.lambtha * x)
        return pdf
