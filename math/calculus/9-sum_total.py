#!/usr/bin/env python3
""" Function that calculates the sum of i² """


def summation_i_squared(n):
    """
    Function that calculate the sum of i²

    Argument:
        n: the stopping condition

    Return:
        the integer value of the sum,
        None if n is not a valid number
    """

    # Check if n is a valid number
    if not type(n) == int and n > 0:
        return None

    # Calculate the sum using the closed-form formula:
    # (n(n+1)(2n+1))/6
    sum_of_squares = (n * (n + 1) * (2 * n + 1)) // 6

    return sum_of_squares
