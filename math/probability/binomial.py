#!/usr/bin/env python3
""" Module containing a class function Binomial that
represents a binomial distribution """


class Binomial:
    """
    class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        class constructor function
        Arguments:
            data: list of the data to be used to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a success
        """

        if data is None:
            # If data is not provided, use n and p directly
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            # Save n as an integer and p as a float
            self.n = int(n)
            self.p = float(p)

        else:
            # If data is provided, calculate n and p from the data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from the data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Calculate p and n from mean and variance
            p = 1 - (variance / mean)
            n = round(mean / p)

            # Recalculate p
            p = mean / n

            # Save n as an integer and p as a float
            self.n = round(n)
            self.p = float(p)

    def factorial(self, n):
        """
        Helper function that calculates the factorial for a given number n > 0
        Argument:
            n: number to factorialize
        Return: the factorial of n
        """
        factorial = 1

        for i in range(1, n + 1):
            factorial *= i
        return factorial

    def pmf(self, k):
        """
        instance method that calculates the value of the PMF for a given
        successes
        Argument:
            k: number of successes
        return: the PMF value
        """

        # Check if k is a non-negative integer
        if k < 0:
            return 0
        # If k is not an integer, convert it to an integer
        if type(k) is not int:
            k = int(k)

        # Calculate the binomial coefficient with the factorial helper function
        binomial_coeff = self.factorial(self.n) / \
            (self.factorial(k) * self.factorial(self.n - k))

        # Calculate the PMF value using the binomial coefficient, p, and (1-p)
        pmf = binomial_coeff * (self.p ** k) * (1 - self.p)**(self.n - k)

        return pmf

    def cdf(self, k):
        """
        Instance method that calculates the value of CDF for a given number
        of successes
        Argument:
            k: number of successes
        Return: the CDF value for k
        """

        # Check if k is a non-negative integer
        if k < 0:
            return 0
        # If k is not an integer, convert it to an integer
        if type(k) is not int:
            k = int(k)

        # Initialize CDF value to zero
        cdf = 0

        # Sum the PMF values for each success up to k
        for i in range(k + 1):
            cdf += self.pmf(i)

        # Return the calculated CDF value
        return cdf
