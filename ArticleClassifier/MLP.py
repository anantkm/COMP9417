#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score

#Reading the training and testset
raw_data_train = pd.read_csv('training.csv')
raw_data_test = pd.read_csv('test.csv')

#Fetching the words and the article labels to train
X_train_temp = raw_data_train['article_words'].to_numpy()
y = raw_data_train['topic'].to_numpy()

#Generating the Count Vector for Training words and transforming.
count = CountVectorizer()
X = count.fit_transform(X_train_temp)

#Splitting the test and training set. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


#Creating MLP Model and training
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

#Predicting Article label with the transformed word count vectors.
predicted_y_mlp = mlp.predict(X_test)

print("----===MLP===----")
# #printPreview(y_test,predicted_y)
# #print(rfc_model.predict_proba(X_test))
print(accuracy_score(y_test, predicted_y_mlp))
print(classification_report(y_test, predicted_y_mlp))

#Computing the Cross Validation Scores for the Model.
scores_mlp = cross_val_score(mlp, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_mlp.mean(), scores_mlp.std() * 2))





