#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:16:57 2019

@author: brgupta
"""

# principle component analysis
# It is one of the most used unsupervised algorithm.
# PCA is used for operation such as visualization ,feature extraction noise filtering ,
# stock market prediction , gene data analysis

# it is used for identify pattern in data
# Detect the correlation between variable
# with PCA will reduce the dimensions of a d-dimentional dataset by projecting it on 
# k -dimentional sub-space ( where k<d)

# PCA is attempting the relationship between X and Y.
# find list of principle axes


# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# fitting logistic regression to the training data set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) # for the same result we put the parameter random_state = 0
classifier.fit(X_train,y_train)

# predicting on the test set, y_red is vector of prediction
y_pred = classifier.predict(X_test)


# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# graphical representation of our predidction
# visualize the training set result
# represented in a graphical formate to get the data prediction
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop =X_set[:,0].max() + 1 , step =0.01),
                    np.arange(start = X_set[:,1].min()-1, stop =X_set[:,1].max() + 1 , step =0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                 c = ListedColormap(('red','green','blue'))(i),label = j)
    
plt.title('Logistic regression training set')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop =X_set[:,0].max() + 1 , step =0.01),
                    np.arange(start = X_set[:,1].min()-1, stop =X_set[:,1].max() + 1 , step =0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                 c = ListedColormap(('red','green','blue'))(i),label = j)
    
plt.title('Logistic regression test set')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
 
