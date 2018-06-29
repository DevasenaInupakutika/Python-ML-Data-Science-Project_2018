#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:13:46 2018

@author: devasenainupakutika
"""

import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

print("Number of missing values per column in df: ", df.isnull().sum())

print("Underlying numpy array of df is: ", df.values)

print("After dropping the rows with missing values: ",df.dropna())

print("Dropping the columns with at least one missing value: ",df.dropna(axis=1))

print("Trying other options in the dropna function for missing data")

print("Only dropping rows where all the columns are NaN: ",df.dropna(how='all'))

print("Dropping rows that haven't at least 4 non-NaN values: ",df.dropna(thresh=4))

print("Only drop rows where NaN appears only in specific columns: ",df.dropna(subset=['C']))