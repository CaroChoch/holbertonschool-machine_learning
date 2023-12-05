#!/usr/bin/env python3
"""function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    function that performs matrix multiplication
        Arguments:
            mat1: first 2D matrix
            mat2: second 2D matrix
        Return:
            a new matrix if all elements are in the same type/shape
            Otherwise : None
    """

    # Check if the number of columns in mat1 is = to the number of rows in mat2
    if len(mat1[0]) == len(mat2):
        # Initialize the result matrix with zeros
        result = [[0] * len(mat2[0]) for row1 in range(len(mat1))]

        # Loop over the rows of mat1
        for row1 in range(len(mat1)):
            # Loop over the columns of mat2
            for col2 in range(len(mat2[0])):
                # Loop over the common dimension :columns of mat1, rows of mat2
                for com_D in range(len(mat2)):
                    # Accumulate the product of corresponding elements
                    result[row1][col2] += mat1[row1][com_D] * mat2[com_D][col2]

        return result  # Return the resulting matrix
    else:
        return None  # Return None if matrices cannot be multiplied
