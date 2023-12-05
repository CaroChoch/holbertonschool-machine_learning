#!/usr/bin/env python3
"""function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """
    function that adds two arrays element-wise
        Arguments:
            arr1: first array
            arr2: second array
        Return:
            a new list of integers or floats if
            arr1 and arr2 are the same shape
            otherwise None
    """
    # Check if the length of arr1 is equal to the length of arr2.
    if len(arr1) == len(arr2):
        # If the arrays are of the same length,
        # initialize an empty list called result.
        result = []

        # Iterate over the indices of the arrays using a for loop.
        for row in range(len(arr1)):
            # Add the corresponding elements of arr1 and arr2 and
            # append the result to the list.
            result.append(arr1[row] + arr2[row])

        # Return the list containing the element-wise sum.
        return result
    else:
        # If the arrays are not of the same length, return None.
        return None
