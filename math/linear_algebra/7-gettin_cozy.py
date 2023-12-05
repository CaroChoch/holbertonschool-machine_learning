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

    # Concatenate along axis 0 (vertical concatenation)
    if axis == 0:
        # Check if the number of columns in both matrices is the same
        if len(mat1[0]) == len(mat2[0]):
            new_matrix = mat1 + mat2
            return new_matrix
        else:
            # Return None if matrices have different number of columns
            return None

    # Concatenate along axis 1 (horizontal concatenation)
    elif axis == 1:
        # Check if the number of rows in both matrices is the same
        if len(mat1) == len(mat2):
            # Iterate over rows and concatenate corresponding rows
            for i in range(len(mat1)):
                new_matrix.append(mat1[i] + mat2[i])
            return new_matrix
        else:
            # Return None if matrices have different number of rows
            return None

    # Return None for invalid axis
    else:
        return None
