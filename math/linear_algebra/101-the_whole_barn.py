#!/usr/bin/env python3
""" function that adds two matrices """


def add_matrices(mat1, mat2):
    """
    function that adds two matrices

    Arguments:
        mat1: the first matrix containing ints/floats
        mat2: the second matrix containing ints/floats

    Return:
        a new matrix if all elements are in the same
        type/shape
        Otherwise: None
    """

    # Check if both mat1 and mat2 are numbers (int or float)
    if ((type(mat1) is int or type(mat1) is float)
       and (type(mat2) is int or type(mat2) is float)):
        return mat1 + mat2

    # Check if both mat1 and mat2 are lists
    elif not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # Check if the number of rows in mat1 is equal to the number
    # of rows in mat2
    if len(mat1) != len(mat2):
        return None

    new_matrix = []
    # Iterate through each row of mat1 and mat2
    for row_nb in range(len(mat1)):
        # Recursively add each element in the current row
        if add_matrices(mat1[row_nb], mat2[row_nb]):
            new_matrix.append(add_matrices(mat1[row_nb], mat2[row_nb]))
        else:
            # If the matrices have different shapes, return None
            return None

    return new_matrix
