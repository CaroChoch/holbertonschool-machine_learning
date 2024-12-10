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
        # list with length of the matrix + the result of the recursive call
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return [len(matrix)]
