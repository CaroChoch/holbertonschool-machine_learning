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

    # Base case: when n is 1, return 1^2
    if n == 1:
        return 1

    # Recursive case: sum of squares up to n is
    # n^2 + sum of squares up to n-1
    return n**2 + summation_i_squared(n-1)
