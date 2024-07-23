#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the CSV file using the from_file function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Select only the 'High', 'Low', 'Close', and 'Volume_(BTC)' columns
# Use iloc with step 60 to select every 60th row, effectively resampling
# the data to hourly intervals
df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]

# Print the last 5 rows of the DataFrame
print(df.tail())
