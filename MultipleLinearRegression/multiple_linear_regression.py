#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:33:00 2019

@author: brgupta
"""


# first step is data preprocessing templates

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# encoding categorical 
# encoding the independent variable
#from sklearn.preprocessing import OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,3] = labelencoder_X.fit_transform(X[:,3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
# first column of the country name has encoded value
X[:,3] = labelencoder_X.fit_transform(X[:,3])
# we are going to create dummy encoding , insteed of having one column of country,
# we are going to create three column. for this we are usng one hot encodded class.
ct = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float)

# avoiding the dummy variable trap
X = X[:,1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
# feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''


# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict on train set
y_pred = regressor.predict(X_test)

# building the optimal model using backward propagation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)
# set the significant value
X_opt = X[:,[0,1,2,3,4,5]]
# creating a siple ordinary Least squares model 
# endog - endogenous response variable
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

# check if p>ls value then remove the independent variable of that column
X_opt = X[:,[0,1,3,4,5]]
# creating a siple ordinary Least squares model 
# endog - endogenous response variable
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()