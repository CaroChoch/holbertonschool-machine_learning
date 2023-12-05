#!/usr/bin/env python3
"""function that concatenates 2 arrays"""


def cat_arrays(arr1, arr2):
    """
    function that concatenates 2 arrays
        Arguments:
            arr1: first array
            arr2: second array
        Return:
            the concatenation of the 2 arrays
    """
    # Concatenate the two arrays using the '+' operator.
    new_array = arr1 + arr2

    # Return the result, which is the concatenated array.
    return new_array
