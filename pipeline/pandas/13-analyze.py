#!/usr/bin/env python3
"""
Calculate descriptive statistics for all columns in pd.DataFrame except
Timestamp
"""


def analyze(df):
    """
    Calculate descriptive statistics for all columns in pd.DataFrame except
    Timestamp
    Arguments:
        - df is the pd.DataFrame to analyze
    Returns:
        The descriptive statistics for all columns in the pd.DataFrame except
        Timestamp
    """
    # calculate descriptive statistics for all columns in pd.DataFrame except
    # Timestamp
    stats = df.drop(columns=['Timestamp']).describe()

    # Return the descriptive statistics
    return stats
