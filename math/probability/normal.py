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
            # If data doesn't contain at least two data points raise ValueError
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # Calculate mean and stddev from the data
                self.mean = float(sum(data) / len(data))
                # The standard deviation is the square root of the variance
                variance = 0
                for i in range(len(data)):
                    # variance is calculated by taking the average of the
                    # squares of the differences of each value from the mean.
                    variance += (data[i] - self.mean) ** 2
                self.stddev = (variance / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
        instance method to calculate the z-score of a given x-value
        argument:
            x: the x-value
        Return: the z-score of x
        """

        # z-score is the number of standard deviations from the
        # mean for a particular value in a statistical distribution
        z_score = (x - self.mean) / self.stddev
        return z_score

    def x_value(self, z):
        """
        Instance method to calculate the x-value of a given z-score
        Argument:
            z: the z-score
        Return: the x-value of z
        """

        x_value = self.mean + z * self.stddev
        return x_value

    def pdf(self, x):
        """
        Instance method that calculates the value of the PDF
        for a given x-value
        Argument:
            x: the x-value
        Return: the PDF value for x
        """

        e_value = 2.7182818285
        pi_value = 3.1415926536

        coeff = 1 / (self.stddev * (2 * pi_value)**0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev)**2

        pdf = coeff * e_value**exponent
        return pdf

    def cdf(self, x):
        """
        instance method that calculates the CDF of a given x-value
        Argument:
            x: the x-value
        Return: the CDF value for x
        """

        pi_value = 3.1415926536

        # step 1: calculation of the argument (arg)
        # for the error function (erf)
        arg = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # step 2: calculation of the approximation of the error function (erf)
        erf_approx = arg - (arg ** 3) / 3 + (arg ** 5) / 10 \
            - (arg ** 7) / 42 + (arg ** 9) / 216
        
        # step 3: calculation of the error function (erf)
        erf = (2 / pi_value ** 0.5) * (erf_approx)

        # calculate cdf
        cdf = 0.5 * (1 + erf)
        return cdf
