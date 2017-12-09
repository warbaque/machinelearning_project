
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

X_train = pd.read_csv("data/train_data.csv", sep=",", header=None)
X_validation = pd.read_csv("data/test_data.csv", sep=",", header=None)
y_train = pd.read_csv("data/train_labels.csv", sep=",", header=None)

X_harjoitus, X_testi, y_harjoitus, y_testi = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2, random_state=12)

X_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1000)).fit_transform(X_harjoitus)
print(X_scaled)
clf = MultinomialNB()
clf.fit(X_scaled, y_harjoitus.values.ravel())

y_pred = clf.predict(X_testi)

print(accuracy_score(y_pred, y_testi))
print(y_pred)
"""
X_scaled = preprocessing.MinMaxScaler().fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_scaled, y_train.values.ravel())
"""
