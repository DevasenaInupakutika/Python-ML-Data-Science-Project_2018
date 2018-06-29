#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:08:51 2018

@author: devasenainupakutika
"""

#Wine dataset analysis consisting of 178 wine samples and 13 features describing their chemical properties

#Reading the wine dataset
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Wine dataset
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

print(df_wine)

#Naming the columns
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))

#Printing first 5 samples
print(df_wine.head())