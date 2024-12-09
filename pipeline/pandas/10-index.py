#!/usr/bin/env python3
""" Set the index to the 'Timestamp' column """


def index(df):
    """
    Set the index to the 'Timestamp' column
    df: pandas DataFrame
    Return: new DataFrame
    """
    # Set the index to the 'Timestamp' column
    df = df.set_index('Timestamp')

    # Return the new DataFrame
    return df
