# Data preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#import the dataset
dataset = pd.read_csv('Data.csv')
# Dataset has 3 independent variable of column country ,Age and salary with 10 observation
X = dataset.iloc[:, :-1].values #all the column except last one
y = dataset.iloc[:,3].values # select last one

# if somewhere we are missing the data then we can remove that data but is quite dangerous.
#Taking care of missing data

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
# first column of the country name has encoded value
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# we are going to create dummy encoding , insteed of having one column of country,
# we are going to create three column. for this we are usng one hot encodded class.
ct = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# here in data age and salary are not in same scale.This will cause in your machine learning model
# this will create problem while calculating eucledean distance.
# first type of feature is standardisation Xstand = X-mean(X) / standard deviation (X)
# second Normalisation Xnorm = X-min(X) / Max(X)-Min(X)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

