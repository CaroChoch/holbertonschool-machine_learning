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
    # Index the pd.DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Include all timestamps from bitstamp up to including timestamp 1417411920
    df2 = df2.loc[:1417411920]

    # Concatenate the pd.DataFrames
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Return the new DataFrame
    return df
