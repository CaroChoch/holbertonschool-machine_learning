#!/usr/bin/env python3
""" Function that slices a DataFrame """


def slice(df):
    """
    Function that flips the switch on a DataFrame
    Return : the new DataFrame
    """
    # Select only the 'High', 'Low', 'Close', and 'Volume_(BTC)' columns
    # Use iloc with step 60 to select every 60th row, effectively resampling
    # the data to hourly intervals
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]

    # Return the new DataFrame
    return df
