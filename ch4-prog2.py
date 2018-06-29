#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:35:42 2018

@author: devasenainupakutika
"""

#Imputing missing values

from sklearn.preprocessing import Imputer
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

imr = Imputer(missing_values='NaN',strategy="mean",axis=0) #axis 0 is for mean value for columns
imr = imr.fit(df)

imputed_data = imr.transform(df.values)
print(imputed_data)