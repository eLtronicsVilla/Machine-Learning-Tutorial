# Machine-Learning-Tutorial
This repository contains all the stuff related to Machine Learning

## Important terms:

##  What are various types of Machine Learning?
1. Supervised Learning:   
Machine is trained on a pre-defined dataset ie. Labeled data to get known output.This training is under external supervision.
Ex- Linear regression,Logistic regression,Support vector machine,KNN,etc.

2. Unsupervised Learning:    
In this model will learn through observation and find structures in data,data is not label here,so will get unknown output.No supervision is required.It will automatically find the pattern and relationship with that dataset by creating cluster.
Ex-K-means,C-means,etc.

3.Reinforcement Learning    
This is hit and trail method.Model will learn on basis of reward and penality.There is no pre-defined data , no supervision.
Ex- Q-learning,SARSA,etc.  

Note : There is one more type Semi-Supervised Learning, where half of the data is labeled and half of the data is not labeled.   

## Difference in Machine Learning and Deep Learning   
Machine Learning is all about the algorithm that parse data , learn from the data and apply what thet have learned to make informed decision.    
Deep Learning is a form of Machine Learning that is inspired by the structure of human brain and it's particularly effective in feature detection.    

## Difference in Classification and Regression    
It is part of supervised learning.Regression predicts a value from the continuous set and classification predict the label to the class.    

Ex- classify a person as Man or Woman, predict the price of a stock over a period of time belong to regression problem.

## What is selection bias?       
Statistical error that causes a bias in the sampling portion of the experiment.It can produce inaccurate conclusion if the selection bias is not identified.     

## What is precision and recall?       
It is the ratio of number of event that you can correctly recall to a number of all correct event.     
Precision is the ratio that the number of event you recall correctly to the total number of your recall.    
Number of event you can correct recall - True positive( they are correct and you recall them)    
Number of all correct event - True positive (They are correct and you recall them) + False negative (They are correct but you don't recall them)     
Number of all event you recall- True Positive(They are correct and you recall them)+false positive (They are not correct but you recall them)     
recall = true positive / (True positive + False negative)     
precision = true positive / ( true positive + false positive)          

## What is confusion matrix?      
Confusion matrix or a error matrix is a table which is used for summarizing the performance of a classification algorithm.

## What is inductive and deductive learning?        
Indunctive Learning - In this first we have some observation on dataset and then model conclusion.    
Deductive Learning - In this we have some model conclusion on dateset first and then observation.    

## How KNN is different from K-means clustering?   
KNN is supervised algorith and K-means is unsupervised algorithm.    
KNN belongs to classification or regression where as K-means belongs to clustering.   
In KNN, k is represented as nearest neighbours used to classify or predict in case of continuous variable/regression.
In K-means, it is the k cluster you are going to identify out of cluster data.     

## What is ROC curve?
ROC represents as Receiver Operating characterstics curve.    
It is a fundamental tool as one of the performance paramter of the model and this is used as the cases of binary classification.It is used in diagnostic test.It is the plot of true positive rate to false positive rate.   

## Difference between typeI and typeII error.   
TypeI error is False Positive,while typeII error is false negative.   

## Difference between Gini Impurity and Entropy in a decision Tree?    
Impurity here, defined as how disclassify your classes within this tree.   
Gini is when you pick a random label out of your samples,It tries to add those probability then create a impurity.
( 1- those probability), the lesser it identifies clusters are labed in different groups.
Entropy is measurement of lack of information.   
Performance wise both are same , but Gini is less computational and overhead.   

## Difference between entropy and information gain?    
Entropy is a indicator of how messy your data. It keeps decreasing when you reaches close to leaf node.    
Both are releavant to each other , as your entrpy will decrease , information gain will increase.    

## What is overfitting and how you will ensure that you are not overfitting with the model?    
Underfitting means our model is not learning completely, what is there with data.It is biased to specific set of data.
It is not learning the pattern which are there with the data.If new data comes in , it would not be able to classify properly.   
In balanced condition , your model is learning within the pattern of the data in a very generalized manner.Generalized means our model is learning on training data and it gives testing data ,where it in the training data.
It is very much closely getting fitted with the data which is there whithin the training set.It's very much learning all the parameter in the training data.   

Way to control overfitting :     
1.Collect more data     
2.Use ensembling method to average the model.    
3.Choose simpler model    

## What is ensamble learning technique?     
Ensamble is nothing but different model ensamble together.Each model is try to understand the data in a different manner.Each model will try to capture a different pattern within the data,These all are weak learners, we combine them together,It's become a better moddel , a single model is created out of it, which is a better predictor model. 
In regression is tries to average those values and create a single value.In case of classification you tries to mejority of all those.    
There are two components in ensamble model.    
1. Bagging 
2. Boosting  

For all samples a single algorithm is used.and at the last you combine all the output and it's called bagging.
In boosting is try to undersatnd the miss-classification which are there in previous model and try to learn those , and gives single model.so each model will learn and become better.   

This will reduce the overfitting and variance of the model.

## What is outlier and what will you do if you find?
You can screen the outlier in different way, one of the way is box-plot.
1.Box-plot uses the formula IQR ( Inter Quartile range)*1.5 +3rd quartile-1st quartile
2.Probabilistic and Statistical model
Using the property of different stastical model, such as normal distribution,exponential distribution. the way your data follows , you can see your model is following those distribution, if the case it flollowing at out of boundary the distribution, you can treat them as outlier.

3.Linear model
With the linear model you can try to see, In time series linear model if you are learning the data.whenever new outlier comes you will try to screen those.FOr logistic regression you can flag them, your model will learn which is the outlier and which is the not.

4.Proximity based model
It is like K-means clustering of the data, so those will fall outside of the cluster , they will form their own cluster and you can see that it has your classified versions of your outlier.

Now , how to handle these classifier?
If you have very high data and if you can drop those outlier, then you can do like that, you can use the percentile.
Impute based on some buisiness rule.The data you can use , using those you can impute your outlier.So that you can go with the model creation.

## What is Co-linearty and multi-colinearty?
Colinearty occurs when two predictor variable in a multi-regression have same correlation.
Multi-colinearty occurs when more than two predictor variable are inter-correlated.

## What is EigenVector and EigenValue?
Eigenvector of a square matrix A is a non-zero vector x , such that for some number (lambda) , we have Ax=(lambda)x
Here (lambda) is eigenvalue.
Eigenvectors are use to understand the linear transformation within your original square matrix.
We calculate it for co-relation and co-variation matrix.
This algorithm can be used in PCA and  factor analysis.It is used for dimention reduction.
When you will apply linear transformation , this eigenvector will give you, in which directions the transformation will be apply.
This is also used in compression as like image compression.

## What is A/B testing?
It is hypothetical statistical testing for randomize experiment with two variable A and B.
how different model perform as compared to each other.Based on different testing A/B testing will use to identify which test is better.
how different functionality are creating the better outcome and revanues.

## What is cluster sampling ?
It is a process of randomly selecting intact group within the defined population.
In cluster sampling we are trying to sample in different clustering sample.
It is a probability sample where each sample unit is collection or cluster of elements.

## Running a binary classification tree is quite easy.How trees decide on which variable to split at the root node and suceed the child node.
Gini-index and entropy can be used to decide which variable is best fit for splitting the decision tree at root node.
Gini calculation on subnode - sum of square of probabily of success and failure.
Calculate Gini for split using weighted Gini score for each node for that split.
Entropy is the measure of impurity or randomness in data.
Entropy = -(p)log(q) - (q)log(p)

## What are preferable library for plotting in python
1.Matplotlib : 
It is used far basic ploting like bar,pies,lines,scatter plots,etc.
2. Seaborn:
It build on top of matplotlib and pandas to ease data ploting.It is used for statistical visualization.
3.Bokeh:
Used for interactive visualization.In case your data is too complex and you haven't find any message in the data.Then use Bokeh to create interactive visualization that will allow your viewer to analyse your data.

## Numpy and Scipy
Numpy is a part of Scipy.
Numpy defines array along with some basic numerical function like indexing , sorting and reshaping.
Scipy implements computations such as numerical integration,optimization and machine learning using numpy's functionalities.

## How can you handle duplicates value in dataset for a variable in python?
use "duplicate()" function to find out duplicated values for a variable and delete them using "drop_duplicates()".


