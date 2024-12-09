#!/usr/bin/env python3
""" Concatenate two pd.DataFrames """

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate two pd.DataFrames
    df1: pd.DataFrame
    df2: pd.DataFrame
    Return: new DataFrame
    """
    # Include all timestamps from bitstamp up to and including timestamp 1417411920
    df2 = df2.loc[:1417411920]

    # Set the index to the 'Timestamp' column
    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')

    # Concatenate the pd.DataFrames
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Return the new DataFrame
    return df
