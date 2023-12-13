#!/usr/bin/env python3
""" function that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """
    function that calculates the integral of a polynomial
    Arguments:
        poly: coefficients representing a polynomial
        C: integer representing the integration constant
    Return:
        a new list of coefficients representing the integral of the polynomial
    """

    # Check if poly is a valid list
    if not isinstance(poly, list):
        return None

    # Check if all coefficients are integers or floats
    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    # Check if poly is not empty
    if not poly:
        return None

    #  checks if the integration constant C is an integer
    if not isinstance(C, int):
        return None

    # If the polynomial is a constant zero,
    # the integral is the integration constant
    if poly == [0]:
        return [C]

    # Initialize the integral list with the integration constant
    integral = [C]

    # Iterate through each coefficient in the polynomial
    for i in range(len(poly)):
        # Checks if the result of the division is an integer
        # (no remainder when divided by 1)
        if (poly[i] / (i + 1)) % 1 == 0:
            # If it's an integer, convert the result to an integer before
            # appending to the integral list
            integral.append(int(poly[i] / (i + 1)))
        else:
            # If it's not an integer, append the result directly
            # to the integral list
            integral.append(poly[i] / (i + 1))

    # Return the list representing the integral of the polynomial
    return integral
