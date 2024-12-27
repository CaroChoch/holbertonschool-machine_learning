#!/usr/bin/env python3
""" Concatenate two dataframes with a MultiIndex """

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    takes two dataframes and concatenates them with a MultiIndex
    Arguments:
        - df1: pd.DataFrame
        - df2: pd.DataFrame
    Returns:
        A new pd.DataFrame with a MultiIndex
    """

    # Filter dataframes to include only rows with 'Timestamp' between
    # 1417411980 and 1417417980
    df1 = df1.loc[
        (df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)]
    df2 = df2.loc[
        (df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)]

    # Index the pd.DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Concatenate the two dataframes along the indexm creating a MultiIndex
    # with keys 'bitstamp' and 'coinbase'
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Swap the levels of the MultiIndex and sort the index to display the rows
    # in chronological order
    df = df.swaplevel(0, 1, axis=0).sort_index()

    # Return the new DataFrame
    return df
