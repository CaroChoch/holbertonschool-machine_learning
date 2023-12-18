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
            elif p < 0 or p > 1:
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
