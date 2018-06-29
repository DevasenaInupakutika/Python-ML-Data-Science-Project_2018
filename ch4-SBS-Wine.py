#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:08:51 2018

@author: devasenainupakutika
"""
#Implementation of SBS algorithm for feature selection
#Wine dataset analysis consisting of 178 wine samples and 13 features describing their chemical properties

#Reading the wine dataset
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from itertools import combinations
import numpy as np
