#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:04:55 2019

@author: brijesh gupta
"""

# decision tree regression model

# first step is data preprocessing templates

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# feature scale your data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# fitting the dicision tree regression model to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
 

# predicting the new result with decision tree regression
#y_pred = regressor.predict(np.array([6.5]))
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

# visualizing the dicision tree regression results ( for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or false (Decision Tree Regression Model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()