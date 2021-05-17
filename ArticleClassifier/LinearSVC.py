#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score


#Reading the training and testsets
raw_data_train = pd.read_csv('training.csv')
raw_data_test = pd.read_csv('test.csv')

#Fetching the words and the article labels to train
X_train_temp = raw_data_train['article_words'].to_numpy()
y = raw_data_train['topic'].to_numpy()


#Generating the Count Vector for Training words and transforming.
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train_temp)

#Splitting the test and training set. 
X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, random_state=0)


#Creating LinearSVC Model and training 
model = LinearSVC()
model.fit(X_train, y_train)

#Predicting Article label with the transformed word count vectors.
predicted_y = model.predict(X_test)

print('accuracy score of LinearSVC', accuracy_score(y_test, predicted_y))
print('precision score of LinearSVC', precision_score(y_test, predicted_y, average='micro'))
print('recall score of LinearSVC', recall_score(y_test, predicted_y, average='micro'))
print('f1 score(micro) of LinearSVC', f1_score(y_test, predicted_y, average='micro'))
print('f1 score(macro) of LinearSVC', f1_score(y_test, predicted_y, average='macro'))


print(classification_report(y_test, predicted_y))

#Computing the Cross Validation Scores for LinearSVC Model.
scores_svc = cross_val_score(model, X_train_counts, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))





