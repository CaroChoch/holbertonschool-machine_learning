#!/usr/bin/env python3
"""function that concatenates two matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    function that that concatenates two matrices along a specific axis
        Argument:
            mat1: the first matrix
            mat2: the second matrix
            axis: he specific axis along which to concatenate (default is 0)
        Return:
            a new numpy.ndarray
    """
    return np.concatenate((mat1, mat2), axis)
