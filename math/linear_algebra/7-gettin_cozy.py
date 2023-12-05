#!/usr/bin/env python3
"""function that concatenates 2 matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function that concatenates 2 matrices along a specific axis
        Arguments:
            mat1: first matrix
            mat2: second matrix
            axis: the specific axis along which to concatenate (default is 0)
        Return:
            a new matrix if all elements are in the same type/shape
            Otherwise : None
    """

    new_matrix = []

    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            new_matrix = mat1 + mat2
            return new_matrix
        else:
            return None

    elif axis == 1:
        if len(mat1) == len(mat2):
            for i in range(len(mat1)):
                new_row = mat1[i] + mat2[i]
                new_matrix.append(new_row)
            return new_matrix
        else:
            return None

    else:
        return None
