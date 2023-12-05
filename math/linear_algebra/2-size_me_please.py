#!/usr/bin/env python3
"""function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """function that calculates the shape of a matrix
        Arguments:
            matrix: matrix to size
        Return:
            a list of integers
    """
    # Check if the first element of the matrix is a list
    if type(matrix[0]) == list:
        # If it is, return a list containing the length of the matrix
        # and the result of the recursive call to the function on the
        # first element of the matrix
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        # If the first element is not a list, simply return a list containing
        # the length of the matrix
        return [len(matrix)]
