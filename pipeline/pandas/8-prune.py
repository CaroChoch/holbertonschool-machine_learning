#!/usr/bin/env python3
""" Remove the entries in the pd.DataFrame where Close is NaN """


def prune(df):
    """
    Remove the entries in the pd.DataFrame where Close is NaN
    Arguments:
        - df is the pd.DataFrame to prune
    Returns:
        The pruned pd.DataFrame
    """

    # Remove the entries in the pd.DataFrame where Close is NaN
    df = df.dropna(subset=['Close'])

    # Return the pruned pd.DataFrame
    return df
