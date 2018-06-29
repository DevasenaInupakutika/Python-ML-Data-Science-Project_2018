#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:21:13 2018

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

#Partitioning dataset into train and test sets
#where X is from columns 1 to 13 and 0th column is class label
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Feature scaling (Normalization and standardization): Normalization is also called min-max scaling

#Min-max scaling procedure
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print("Training set normalized features: ",X_train_norm)
print("Test set normalized features: ",X_test_norm)

#Standardization Procedure
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print("Training set standardized features: ",X_train_std)
print("Test set standardized features: ",X_test_std)

#A Visual example
ex = np.array([0, 1, 2, 3, 4, 5])

#Standardize
print('standardized:', (ex - ex.mean()) / ex.std())

#Normalise
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

#Using the regularized models from scikit learn
LogisticRegression(penalty="l1")
#Applying l1 penalty to the standardised Wine data above, l1 regularized logistic regression would yield sparse olutions
lr = LogisticRegression(penalty="l1",C=0.1) #With C we control the large error penalties
lr.fit(X_train_std,y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

#Intercepts to check the One-vs-rest approach by Logistic regression due to multi-class labels
print(lr.intercept_)
print("Number of non-zero elements of the weight array: ",lr.coef_[lr.coef_!=0].shape)
print("The weight array is as follows: ",lr.coef_)
#Each row in the weight vector indicates for each class label

#Plotting regularization path i.e. weight coefficients of the different features for different regularization strengths
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('images/04_07.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)
plt.show()

