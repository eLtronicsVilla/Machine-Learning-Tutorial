#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:19:55 2019

@author: brijesh gupta

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

# fitting regression model to the dataset

# here you can  

# predicting the new result with polynomial regression
y_pred = regressor.predict(6.5)

# visualizing the regression results ( for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or false (Regression Model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()