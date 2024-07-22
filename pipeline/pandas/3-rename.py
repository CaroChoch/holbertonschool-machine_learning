#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the  CSV file using the from_file function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename the column 'Timestamp' to 'Datetime'
df = df.rename(columns={'Timestamp': 'Datetime'})

# Convert the timestamp values in 'Datetime' column to datetime objects
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Display only the 'Datetime' and 'Close' columns
df = df[['Datetime', 'Close']]

# Print the last 5 rows of the DataFrame df
print(df.tail())
