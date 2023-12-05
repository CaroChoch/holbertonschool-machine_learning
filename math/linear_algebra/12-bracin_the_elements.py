#!/usr/bin/env python3
"""function that performs element-wise addition, subtraction,
multiplication, and division"""


def np_elementwise(mat1, mat2):
    """
    function that performs element-wise addition, subtraction,
    multiplication, and division
        Argument:
            mat1: the first matrix
            mat2: the second matrix
        Return:
            tuple containing the element-wise sum,
            difference, product, and quotient, respectively
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
