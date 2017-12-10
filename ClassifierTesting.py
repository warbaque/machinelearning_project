import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


classifiers = (
    ("kneighbors",          KNeighborsClassifier()),
    ("decision_trees",      DecisionTreeClassifier(max_depth=25)),
    ("linear_svc",          LinearSVC(random_state=0)),
    ("gaussian",            GaussianNB()),
    ("logistic_regression", LogisticRegression())
)


X_train = pd.read_csv("data/train_data.csv", sep=",", header=None)
X_validation = pd.read_csv("data/test_data.csv", sep=",", header=None)
y_train = pd.read_csv("data/train_labels.csv", sep=",", header=None)

train, test, t_train, t_test = train_test_split(X_train, y_train, test_size=0.3)

for s, clf in classifiers:
    clf.fit(train, t_train.values.ravel())
    print("{:20}:".format(s), clf.score(test, t_test))
