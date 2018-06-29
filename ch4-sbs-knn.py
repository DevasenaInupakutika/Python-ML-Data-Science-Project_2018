#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:22:24 2018

@author: devasenainupakutika
"""

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from SBS import SBS

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

#SBS in action using KNN classifier
knn = KNeighborsClassifier(n_neighbors = 2)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()

#List of features that yielded the good performance on the validation dataset
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

#Evaluating performance of the knn classifier on the original test set
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

#On the selected five feature test set
knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))


