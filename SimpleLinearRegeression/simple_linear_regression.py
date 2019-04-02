  # -*- coding: utf-8 -*-
"""
This is the tutorial for extracting the dataframe from the dictionary object
"""

# first step is data preprocessing tempplates

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Employee_salary.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''
# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

# fitting simple linear regression to the training dataset 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict the salary
y_pred = regressor.predict(X_test)

# visualizing the training set result

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('salary prediction-( train set)')
plt.xlabel('year of experience')
plt.ylabel('salary')

# visualizing the test set result
plt.scatter(y_train,y_test,color='blue')
plt.plot(X_test,regressor.predict(X_test))
plt.label('predict the alary - testdata')
plt.xlabel('year of experience')
plt.ylabel('salary')
