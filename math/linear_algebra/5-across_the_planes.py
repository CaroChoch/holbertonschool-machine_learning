#!/usr/bin/env python3
"""function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """
    function that adds two matrices element-wise
        Arguments:
            mat1: first array
            mat2: second array
        Return:
            a new matrix if mat1 & mat2 are the same type/shape
            otherwise None
    """
    # Check if the matrices have the same number of rows and columns.
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        # Initialize a new matrix with element-wise sums.
        new_matrix = [
            [mat1[row][column] + mat2[row][column]
             for column in range(len(mat1[0]))]
            for row in range(len(mat1))
        ]
        # Return the new matrix.
        return new_matrix
    else:
        # If matrices have different shapes, return None.
        return None
