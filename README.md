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
