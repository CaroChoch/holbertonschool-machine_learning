#!/usr/bin/env python3
"""function that slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """
    function that slices a matrix along specific axes
        Arguments:
            matrix: a numpy.ndarray matrix
            axes: dictionary where the key is an axis to slice along
            value: a tuple representing the slice to make along that axis
        Return:
            a new numpy.ndarray matrix if all elements are in the same
            type/shape
            Otherwise : None
    """

    # Create a copy of the original matrix to avoid modifying the input
    new_matrix = matrix

    # Initialize an empty list to store slices for each axis
    list_slices = []

    # Iterate through the provided axes and apply the slices
    for key in range((max(axes.keys()) + 1)):
        # Get the slice tuple corresponding to the current axis
        value = axes.get(key)

        # Check if a slice is provided for the current axis
        if value:
            # Append a slice object created from the tuple (start, stop, step)
            # to the list_slices
            list_slices.append(slice(*value))
        else:
            # If no slice is provided, append a slice object representing the
            # entire axis (slice(None))
            list_slices.append(slice(None))

    # Create a tuple of slices and apply them to the matrix
    tuple_slices = tuple(list_slices)
    return new_matrix[tuple_slices]
