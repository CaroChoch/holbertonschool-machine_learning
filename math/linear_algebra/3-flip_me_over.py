#!/usr/bin/env python3
"""function that returns the transpose of a 2D matrix, matrix"""


def matrix_transpose(matrix):
    """
    Function that that returns the transpose of a 2D matrix, matrix
    Arguments:
        matrix: matrix to transpose
        Return: a list of integers
    """
    # Use list comprehension to create the transposed matrix
    # Iterate over columns of the original matrix
    # For each column, create a new row in the transposed matrix
    # Each element in the new row is taken from the corresponding row
    # in the original matrix
    transposed_matrix = [
        [matrix[row][column] for row in range(len(matrix))]
        for column in range(len(matrix[0]))
    ]
    return transposed_matrix
