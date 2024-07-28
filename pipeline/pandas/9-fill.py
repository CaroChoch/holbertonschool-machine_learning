#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

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

print(df.head())
print(df.tail())
