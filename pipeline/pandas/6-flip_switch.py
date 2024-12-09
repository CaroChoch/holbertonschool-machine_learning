#!/usr/bin/env python3
""" Function that transposes a DataFrame """


def flip_switch(df):
    """
    Function that transposes a DataFrame
    df: pandas DataFrame
    Return: new DataFrame
    """
    # Transpose the DataFrame
    df = df.T

    # Sort the data in reverse chronological order
    df = df.sort_index(axis=1, ascending=False)

    # Return the new DataFrame
    return df
