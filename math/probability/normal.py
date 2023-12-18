#!/usr/bin/env python3
""" Module containing a class Normal that represents a normal distribution """


class Normal:
    """
    Class representing a Normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor that represents a normal distribution
        Arguments:
            data: list of the data to be used to estimate the distribution
            mean: the mean of the distribution
            stddev: the standard deviation of the distribution
        """

        if data is None:
            # If data is not given, use the provided mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            # If data is given, calculate mean and stddev from the data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # Calculate mean and stddev from the data
                self.mean = float(sum(data) / len(data))
                self.stddev = float((sum((x - self.mean) ** 2 for x in data) /
                                     (len(data) - 1)) ** 0.5)
