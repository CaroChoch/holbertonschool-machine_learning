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

    def factorial(self, n):
        """
        helper function that calculates the factorial of a non-negative int
        Argument:
            n: a non-negative integer
        Returns:
            The factorial of n
        """
        if n == 0:
            return 1
        result = 1
        for i in range(1, n+1):
            result *= i
        return result

    def pmf(self, k):
        """
        Instance method that calculates the value of the PMF
        for a given number of successes
        Argument:
            k: the number of successes
        Return:
            the PMF value for k
        """

        # Ensure k is an integer
        if not isinstance(k, int):
            k = int(k)
        # Check if k is a non-negative value
        if k < 0:
            return 0

        # Value of the mathematical constant e
        e_value = 2.7182818285

        # Calculate the PMF value using the Poisson distribution formula
        pmf_value = (self.lambtha**k * e_value**(-self.lambtha) /
                     self.factorial(k))

        return pmf_value

    def cdf(self, k):
        """
        instance method that calculates the value of CDF for a given
        number of successes
        Argument:
            k: the number of successes
        Return: the CDF value for k
        """

        # Ensure k is an integer
        if not isinstance(k, int):
            k = int(k)
        # Check if k is a non-negative value
        if k < 0:
            return 0

        # Initialize the cumulative distribution function (CDF) value to 0
        cdf_value = 0
        # Loop through each integer i from 0 to k (inclusive)
        for i in range(k + 1):
            # For each i, add the PMF value for i to the CDF
            cdf_value += self.pmf(i)

        # Return the calculated CDF value for the given number of successes k
        return cdf_value
