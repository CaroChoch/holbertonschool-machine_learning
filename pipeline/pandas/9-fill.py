#!/usr/bin/env python3
""" Fill missing values in the Bitcoin dataset """


def fill(df):
    """
    Fill missing values in the Bitcoin dataset
    df: pd.DataFrame
    Return: pd.DataFrame
    """

    # Remove the columns 'Weighted_Price'
    df.drop(columns=['Weighted_Price'], inplace=True)

    # Set missing values in the 'Close' column to the previous row value
    df['Close'] = df['Close'].ffill()

    # Set missing values in High, Low, Open to the same row value of Close
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    df[['Volume_(BTC)', 'Volume_(Currency)']] = df[[
        'Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

    # Return the new DataFrame
    return df
