#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:37:31 2019

@author: brgupta
"""

# support vctor machine : linear classifier
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
classifier = SVC(kernel = 'linear' ,random_state = 0)  # let's use linear kernel , in order to all get the same result , use random_state = None
classifier.fit(X_train,y_train)


# predicting on the test set, y_red is vector of prediction
y_pred = classifier.predict(X_test)


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


