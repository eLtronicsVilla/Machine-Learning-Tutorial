#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:36:15 2019

@author: brgupta
"""

# Randome tree forest regression method

# Step 1: Pick a K random data point from the Training set
# step 2: Build the decision tree associated to these data point
# step 3: Choose a number Ntree of tree you want to build
# Step 4: FOr a data point , make each one of your Ntree trees predict the 
# value of Y to for the data point in question.
# Assign the data point all the average accross the all predicted Y value.

# It is the non-continuation of regression model

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



# fitting Random forest regression model to the dataset

from sklearn.ensemble import RandomForestRegressor
# Here you can choose correct value of n_estimator
regressor = RandomForestRegressor(n_estimators= 300,random_state=0)
regressor.fit(X,y)

# predicting the new result with regression
#y_pred = regressor.predict(6.5)
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

# visualizing the regression results ( for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or false (Regression Model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()