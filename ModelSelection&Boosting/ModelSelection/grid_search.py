#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:14:38 2019

@author: brgupta
"""

# greed search technique

# this technique is using for model performance 
# choosing the correct value of the hyperparameter for model performance increament
# this greed search technique help you for finding the optimal value for the hyperparameter

# How do I know which model to choose for my machine learning problem

# first we should know my problem is regression or classification or clustering probelm.For this you have to 
# look at your dependent variable ,if you don't have a dependent varaible then it's a clustering problem.
# if dependent variable and it's come continuous outcome then your probelm is regression and if it is catagorical outcome then your probelm is classification.

# here grid search will choose if it is SVM model or Kernel-SVM model.


# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# making the confusion matrix
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf' ,random_state = 0)  # let's use linear kernel , in order to all get the same result , use random_state = None
classifier.fit(X_train,y_train)


# grid search can be apply after the model to your training set 
# apply grid search to find best model and best parameter 
from sklearn.model_selection import GridSearchCV
parameters = [{'c':[1,10,100,1000],'kernel':['linear']},{'c':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]}] # key identifier will give you the specific value , for each of it we will give several values
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10, n_jobs= -1) # grid is going to use at least one performacne parameter , either it's accuracy or recall 
grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_


# predicting on the test set, y_red is vector of prediction
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# applying K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X = X_train,y=y_train, cv=10) # most of the time we need 10 fold cross validation
 # if you are using a large data set then you can use a parameter called n_jobs that will use all your CPU for faster execution.
accuracies.mean() # it will give the accuracy for 10 set of test .
accuracies.std() # it will give the standard deviation of the accuracy vector 

# graphical representation of our predidction
# visualize the training set result
# represented in a graphical formate to get the data prediction
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop =X_set[:,0].max() + 1 , step =0.01),
                    np.arange(start = X_set[:,1].min()-1, stop =X_set[:,1].max() + 1 , step =0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                 c = ListedColormap(('red','green'))(i),label = j)
    
plt.title('SVM training set')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()

X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop =X_set[:,0].max() + 1 , step =0.01),
                    np.arange(start = X_set[:,1].min()-1, stop =X_set[:,1].max() + 1 , step =0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                 c = ListedColormap(('red','green'))(i),label = j)
    
plt.title('SVM test set')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()


