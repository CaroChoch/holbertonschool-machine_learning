#!/usr/bin/env python3

import pandas as pd


def array(df):
    """
    Convert a pd.DataFrame to a np.ndarray
    Arguments:
        - df is the pd.DataFrame to convert
    Returns:
        The np.ndarray representation of df
    """

    # Select only the 'High' and 'Close' columns from the DataFrame
    df = df[['High', 'Close']]

    # Get the last 10 rows of the DataFrame
    df = df.tail(10)

    # Convert the DataFrame to a NumPy array
    A = df.to_numpy()

    # Return the NumPy array
    return A
