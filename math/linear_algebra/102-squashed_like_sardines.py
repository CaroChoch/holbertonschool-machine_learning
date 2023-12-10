#!/usr/bin/env python3
""" function that concatenates two matrices along a specific axis"""

def cat_matrices(mat1, mat2, axis=0):
    """
    function that concatenates two matrices along a specific axis

    Arguments:
        mat1: the first matrix containing ints/floats
        mat2: the second matrix containing ints/floats
        axis: the specific axis along which to concatenate (default is 0)

    Return:
        a new matrix if all elements are in the same
        type/shape
        Otherwise: None
    """

    # Initialize an empty list to store the concatenated matrix
    new_matrix = []

    # Check if concatenation is along axis 0
    if axis == 0:
        # Concatenate rows of mat1
        for row_nb in range(len(mat1)):
            new_matrix.append(mat1[row_nb])
        # Concatenate rows of mat2
        for row_nb in range(len(mat2)):
            new_matrix.append(mat2[row_nb])
        return new_matrix
    # Check if matrices have different lengths along the specified axis
    elif len(mat1) != len(mat2):
        return None
    else:
        # Concatenate along other axes recursively
        for row_nb in range(len(mat1)):
            # Check if the concatenated matrices along the axis - 1 have the same type/shape
            if cat_matrices(mat1[row_nb], mat2[row_nb], axis - 1) is None:
                return None
            # Recursively concatenate along the axis - 1
            new_matrix.append(cat_matrices(mat1[row_nb], mat2[row_nb], axis - 1))
        return new_matrix
