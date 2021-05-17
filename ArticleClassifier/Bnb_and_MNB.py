#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score


# READING TRAINING DATA
raw_data_train = pd.read_csv('training.csv')

# READING TESTING DATA
raw_data_test = pd.read_csv('test.csv')

#Fetching Words to train
X_train_temp = raw_data_train['article_words'].to_numpy()

#Fetching Article Labels
y = raw_data_train['topic'].to_numpy()

# Forming the count vectors for the input words and transforming
count = CountVectorizer()
X = count.fit_transform(X_train_temp)

# Splitting the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


#BERNOULLI NAIVE BAYES MODEL GENERATION AND TRAINING
model_1 = BernoulliNB()
model_1.fit(X_train, y_train)
predicted_y_b = model_1.predict(X_test)

print('---BernoulliNB---')
print('accuracy score of Random Forest', accuracy_score(y_test, predicted_y_b))
print('precision score of Random Forest', precision_score(y_test, predicted_y_b, average='micro'))
print('recall score of Random Forest', recall_score(y_test, predicted_y_b, average='micro'))
print('f1 score(micro) of Random Forest', f1_score(y_test, predicted_y_b, average='micro'))
print('f1 score(macro) of Random Forest', f1_score(y_test, predicted_y_b, average='macro'))

print(classification_report(y_test, predicted_y_b))


#MULTINOMIAL NAIVE BAYES MODEL GENERATION AND TRAINING
print('---MultinomialNB---')
model_2 = MultinomialNB()
model_2.fit(X_train, y_train)
predicted_y_m = model_2.predict(X_test)

print('accuracy score of Random Forest', accuracy_score(y_test, predicted_y_m))
print('precision score of Random Forest', precision_score(y_test, predicted_y_m, average='micro'))
print('recall score of Random Forest', recall_score(y_test, predicted_y_m, average='micro'))
print('f1 score(micro) of Random Forest', f1_score(y_test, predicted_y_m, average='micro'))
print('f1 score(macro) of Random Forest', f1_score(y_test, predicted_y_m, average='macro'))

print(classification_report(y_test, predicted_y_m))

#CHECKING THE CROSS VALIDATION SCORES.
scores_bnb = cross_val_score(model_1, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_bnb.mean(), scores_bnb.std() * 2))


scores_mnb = cross_val_score(model_2, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_mnb.mean(), scores_mnb.std() * 2))





