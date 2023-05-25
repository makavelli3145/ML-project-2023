# -*- coding: utf-8 -*-
"""
ML project 2023
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Reading in the data points
training_data = open('traindata.txt', 'r')
data_string = training_data.read()
full_data = np.array(data_string.split(','))[:-1].astype(np.float64).reshape((1100,5))
print("full_data has ", full_data.shape[0], " data points with ", full_data.shape[1], "features")
# Reading in the 4 different labels for each data point
training_labels_file = open('trainlables.txt', 'r')
labels_string = labels_file.read()
full_y_values = np.array(labels_string.split(','))[:-1]
full_y_values = np.where(full_y_values == 'True', True, False).reshape((1100,4))
print("full_y_values has labels for ", full_y_values.shape[0], " data points with ", full_y_values.shape[1], "labels per data points")
 
    
'''
TODO-1: Convert the training_data and training_lables into pandas dataframes
'''

    
'''
TODO-2: Create an SK learn classifier, then train our model
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)