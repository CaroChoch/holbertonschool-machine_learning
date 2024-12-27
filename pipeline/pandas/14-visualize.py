#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the column Weighted_Price
df = df.drop(columns=['Weighted_Price'], axis=1)

# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df = df.set_index('Date')

# Set the missing values in Close to the previous row value
df['Close'] = df['Close'].ffill()

# Set the missing values in High, Low, Open to the same row value of Close
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Set the missing values in Volume_(BTC) and Volume_(Currency) to 0
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[[
    'Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

# Select the data from 2017 and beyond
df = df[df.index >= '2017-01-01']

# Resample the data to daily frequency
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Create a new DataFrame for plotting
df_plot = pd.DataFrame()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()

# Plot the data with compatibility for x-axis
df_plot.plot(x_compat=True)

# Add a title to the plot
plt.title('Cryptocurrency Market Data Over Time')

# Display the plot
plt.show()

# Save the plotted image in the current directory
plt.savefig('visualize.png')
