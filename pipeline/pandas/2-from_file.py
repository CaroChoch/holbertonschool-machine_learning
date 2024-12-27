#!/usr/bin/env python3
""" Load data from a file """
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame

    Arguments:
        - filename is the file to load from
        - delimiter is the column separator
    Returns: the loaded pd.DataFrame
    """
    # Load the data
    df = pd.read_csv(filename, delimiter=delimiter)

    # Return the DataFrame
    return df
