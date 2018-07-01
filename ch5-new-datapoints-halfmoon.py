#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:45:16 2018

@author: devasenainupakutika
"""

from rbf_kernel_pca_eigenvalues import rbf_kernel_pca_eigenvalues
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

#Creating half moon dataset and project it on one-dimensional subspace with the 
#updated rbf kernel
X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca_eigenvalues(X, gamma=15, n_components=1)

#Assuming that the 26th point from the half-moon dataset is the new data point x' and below 
#is the task to project it onto this new subspace
x_new = X[25]
x_new
x_proj = alphas[25] # original projection
x_proj

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# projection of the "new" datapoint
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj

plt.scatter(alphas[y == 0, 0], np.zeros((50)),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)

plt.tight_layout()
plt.show()


