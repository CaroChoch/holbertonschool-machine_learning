#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the CSV file using the from_file function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Select only the 'High' and 'Close' columns from the DataFrame
df = df[['High', 'Close']]

# Get the last 10 rows of the DataFrame
df = df.tail(10)

# Convert the DataFrame to a NumPy array
A = df.to_numpy()

# Print the numpy array
print(A)
