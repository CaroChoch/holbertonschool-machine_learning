#!/usr/bin/env python3
""" Create a pd.DataFrame from a np.ndarray """
import pandas as pd


def from_numpy(array):
    """
    Create a pd.DataFrame from a np.ndarray
    Arguments :
        - array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns:
        - the newly created pd.DataFrame
    """

    # Number of columns
    n_cols = array.shape[1]

    # Generate column labels
    columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # Create DataFrame
    df = pd.DataFrame(array, columns=columns[:n_cols])

    return df
