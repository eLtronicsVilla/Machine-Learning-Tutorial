#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:59:45 2019

@author: brgupta
"""

# import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distance')
plt.show()


# fitting hierarchical clustering to the fitting dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage='ward') # ward to minimize the cluster for linkage
y_hc = hc.fit_predict(x) 

# visualising the cluster
plt.scatter(x[y_hc == 0,0],x[y_hc == 0,1],s=100,c='red',label='Cluster 1:Careful')
plt.scatter(x[y_hc == 1,0],x[y_hc == 1,1],s=100,c='blue',label='Cluster 2:standard')
plt.scatter(x[y_hc == 2,0],x[y_hc == 2,1],s=100,c='green',label='Cluster 3: target')
plt.scatter(x[y_hc == 3,0],x[y_hc == 3,1],s=100,c='magenta',label='Cluster 4:careless')
plt.scatter(x[y_hc == 4,0],x[y_hc == 4,1],s=100,c='yellow',label='Cluster 5:sensible')
plt.title('Clusters of client')
plt.xlabel('Anual income (k$))')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()