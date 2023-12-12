#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """
    Function that calculates the derivative of a polynomial

    Argument:
        poly: list of coefficients representing a polynomial

        Return:
            a new list of coefficients representing the
            polynomial
            Or None if the derivative is O
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

    # Calculate the derivative
    # i is the index of the list representing the power of
    # x that the cofficient belongs to
    derivative = [i * poly[i] for i in range(1, len(poly))]

    # If the derivative is 0, return [0]
    if all(coeff == 0 for coeff in derivative):
        return [0]

    return derivative
