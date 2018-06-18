#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:31:26 2018

@author: devasenainupakutika
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from Perceptron import Perceptron
from AdalineGD import AdalineGD
from AdalineSGD import AdalineSGD

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    print("Cmap is: ", cmap)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


#Loading the Iris dataset directly from the UCI repository into a dataframe object
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

#Extracting the first 100 class labels and first (0 index) and third (2 index) features (or column names)
#Printing 4 columns or features hence for now
#Converting the class labels corresponding to Iris-Setosa as -1 and Iris-Versicolor as 1
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1) #Similar to ternary conditional operator, second parameter is when condition is true and third when condition is false
X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
plt.xlabel('Sepal Length [cm]: 1st feature')
plt.ylabel('Petal Length [cm]: 3rd feature')
plt.legend(loc='upper left')
plt.show()

#Train the perceptron algorithm on the above extracted dataset
#Also plots the misclassifications for each epoch to see if the algorithm converged and found a decision boundary that separates the 2 Iris flower classes
ppn = Perceptron(eta=0.1,n_iter=10)
#X is the input and y is output class labels
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs (or iterations)')
plt.ylabel('Number of misclassifications')
plt.show()

#If the plot is at 0.00 already, this means that the perceptron is already converged from the first epoch and is able to 
#classify the training samples perfectly
#2D Convenience function to visualize the decision boundaries for 2D datasets
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()

#Adaline Model Below (Adaptive Linear Neural Model)
#Plot of cost vs number of epochs for 2 different learning rates
#Learning rate 0.01
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

#Learning rate 0.0001
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
plt.show()

#Gradient descent being benefitted by feature scaling but for now, standardization where data is given the property of 
#standard normal distribution with SD of feature column as 1 and the mean of each feature is centered at 0.
# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#Training using Adaline model again
ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
plt.show()

#Adaline learning model with stochastic gradient descent, shuffle and computing weights and updating them after every sample
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
#ada.fit(X_std, y)
#for online learning replacing with partial_fit
ada.partial_fit(X_std[0, :], y[0])

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('./adaline_4.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()
