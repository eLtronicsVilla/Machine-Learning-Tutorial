#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:43:02 2019

@author: Brijesh Gupta

"""

# support vector regression

'''
SVR is a type of support vector machine that supports linear and non-linear 
regression.SVR perform linear regression in higher dimentional space.
In training each of data point represent it's own dimention.

When you are evaluate your kernel between a test point and a point in the
training set the resulting value give you the co-ordinate of your test point,
in that dimention.It require a trianing set T = {X,Y} which covers the domain
of interest and is accompanied by solution of that domain.

The work of the SVM is to approximate the function we used to generate the
training set.

F(X) = Y

In a classification problem , the vector X used to define a hyperplane that
separates the two different classes in your solution.

These vector are used to perform linear regression. The vector closest to the 
test point are reffered to as support vector.
We can evaluate our function anywhere so any vectors could be closest to our 
test evaluation location.
 
'''

# first step is data preprocessing templates

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# feature scale your data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(sc_y.fit_transform(y),dtype=np.float)

# fitting SVR regression model to the dataset
# here you can write code for your regression 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


# predicting the new result with SVR regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
# visualizing the SVR results ( for higher resolution and smoother curve)

plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or false (Support Vector Regression Model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or false SVR Model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
