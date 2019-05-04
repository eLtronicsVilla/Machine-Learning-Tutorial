#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 12:03:57 2019

@author: brgupta
"""

# XGBoot -  Install XGBoost in your system

# this model is popular if you work on large dataset and high accuracy.
# XGBoot has high execution and fast speed, XGBoost is popular for High performance,
# fast execution speed, you can keep all the interpretation of your model.

# import the library

import numpy as np #for all numerical and computational operation
import matplotlib.pyplot as plt  # to plot the graph
import pandas as pd  # library to work on datasheet  

# Importing the dataset ie geography and gender
dataset = pd.read_csv('Churn_Modelling.csv')  #this is to read the csv file using pandas libraary
X = dataset.iloc[:, 3:13].values  # collecting the row value from range 1 to 12 
y = dataset.iloc[:, 13].values  # collecting the row value of 13

# Encoding categorical data of dependent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  # encoding the variable  
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # label 0,1,2 for France,Spain,Germany
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # label 0 or 1 for male or female
onehotencoder = OneHotEncoder(categorical_features = [1]) # to create dummy variable to the country
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set 80% and Test set 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# XGBoost is a gradiant boost technique 
# fitting XGBoots to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# predict true or false output 
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# applying K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X = X_train,y=y_train, cv=10) # most of the time we need 10 fold cross validation and will get 10 accuracy. 
 # if you are using a large data set then you can use a parameter called n_jobs that will use all your CPU for faster execution.
accuracies.mean() # it will give the accuracy for 10 set of test .
accuracies.std() # it will give the standard deviation of the accuracy vector 
