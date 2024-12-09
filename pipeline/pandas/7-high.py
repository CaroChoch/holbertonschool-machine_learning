#!/usr/bin/env python3
""" Sort the data in descending order of the 'High' order """


def high(df):
    """
    Sort the data in descending order of the 'High' order
    df: pandas DataFrame
    Return: new DataFrame
    """
    # Sort the data in descending order of the 'High' order
    df = df.sort_values(by='High', ascending=False)

    # Return the new DataFrame
    return df
