#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:22:28 2018

@author: devasenainupakutika
"""

#Handling categorical data (Data example for understanding)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([['green','M',10.1,'class1'],['red','L',13.5,'class2'],['blue','XL',15.3,'class1']])

df.columns = ['color','size','price','classlabel']

print(df)

#Mapping the ordinal features
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

#Encoding class labels
#Similar approach to mapping of ordinal features 
#Also class labels are not ordinal
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

#Using the mapping dictionary to transform the class labels into integers
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

#Converting class labels back to the original string representations
inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

#Inverse mapping using scikit-learn's labelencoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

#Using inverse_tranfrom to convert integer class labels back to string representations
print(class_le.inverse_transform(y))

#Performing one-hot encoding on nominal features
X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0]) #For color column
print(X)

#Performing one-hot encoding on nominal features  (Dummy feature for every unique feature)
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

#More convenient way of one-hot encoding using get_dummies. get_dummies gets applied only to string columns
print(pd.get_dummies(df[['price','color','size']]))