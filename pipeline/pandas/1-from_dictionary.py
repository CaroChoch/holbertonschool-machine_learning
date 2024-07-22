#!/usr/bin/env python3
""" Script that creates a pd.DataFrame from a dictionary """
import pandas as pd


# Dictionary
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Row labels
row_labels = ['A', 'B', 'C', 'D']

df = pd.DataFrame(data, index=row_labels)
