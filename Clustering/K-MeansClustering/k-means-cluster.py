#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:12:56 2019

@author: brgupta
"""

# K-means cluster 

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the mall dataset with pandas
dataset=pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# importing the mail dataset with pandas
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
   kmeans = KMeans(n_clusters =i ,init='k-means++',max_iter=300,n_init=10,random_state=0)
   kmeans.fit(x)
   wcss.append(kmeans.inertia_)
   
plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('number of clusters')
#plt.y_lable(wcss)
plt.show()


# applying K-means to the mall dataset
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means = kmeans.fit_predict(x)
plt.scatter(x[y_means == 0,0],x[y_means == 0,1],s=100,c='red',label='Cluster 1:Careful')
plt.scatter(x[y_means == 1,0],x[y_means == 1,1],s=100,c='blue',label='Cluster 2:standard')
plt.scatter(x[y_means == 2,0],x[y_means == 2,1],s=100,c='green',label='Cluster 3: target')
plt.scatter(x[y_means == 3,0],x[y_means == 3,1],s=100,c='magenta',label='Cluster 4:careless')
plt.scatter(x[y_means == 4,0],x[y_means == 4,1],s=100,c='yellow',label='Cluster 5:sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='Centroids')
plt.title('Clusters of client')
plt.xlabel('Anual income (k$))')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
