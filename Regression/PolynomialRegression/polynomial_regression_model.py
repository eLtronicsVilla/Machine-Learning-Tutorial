#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:05:50 2019

@author: brijesh Gupta
"""

# first step is data preprocessing templates

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('position_salaries.csv')
X = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

'''
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
X = X[:,1:]'''

# No need to apply feature scaling here.
# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X,y)
# fitting polynomial regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_polyReg = poly_regressor.fit_transform(X)
lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_polyReg,y)

# visualising the Linear regression result
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_regressor.predict(X),color = 'blue')
plt.title('Truth or false (Linear regression)')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()

# visualizing the polynomial regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,lin_regressor2.predict(poly_regressor.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or false (Polynomial regression)')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()

# predicting the new result with linear regression
lin_regressor.predict(6.5)
# prediction with the new result with polynomial regression
lin_regressor2.predict(poly_regressor.fit_transform(6.5))