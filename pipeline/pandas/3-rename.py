#!/usr/bin/env python3
""" Rename the column Timestamp to Datetime and convert the values in the
    'Datetime' column to datetime objects """

import pandas as pd


def rename(df: pd.DataFrame):
    """
    Rename the column Timestamp to Datetime and convert the values in the
    'Datetime' column to datetime objects
    Arguments:
        - df is the pd.DataFrame to modify
    Returns:
        The modified pd.DataFrame
    """

    # Rename the column 'Timestamp' to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the timestamp values in 'Datetime' column to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Display only the 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
