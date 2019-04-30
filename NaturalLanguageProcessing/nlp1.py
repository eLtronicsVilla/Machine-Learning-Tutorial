#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:55:43 2019

@author: brgupta
"""

# natuaral language processing

# import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting =3)

# Cleaning the Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# corpus is a collection of text
corpus =[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i]) # we do'nt want to remove these characters
    review = review.lower()
    review = review.split()
    # finding the root word 
    ps = PorterStemmer()
    # create object of this classs
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# creating the bag of word model

from sklearn.feature_extraction.text import CountVectorizer
# we have to remove the stop_words from the corpus
cv = CountVectorizer(max_features = 1500)
# apply fit_transformation mthod to create huge matix of feature to train the review .
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# train the data to create machine learning model efficiently
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# fitting  the training data set
# here your code of classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# predicting on the test set, y_red is vector of prediction
y_pred = classifier.predict(X_test)


# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

