import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


classifiers = (
    ("kneighbors",          KNeighborsClassifier()),
    ("decision_trees",      DecisionTreeClassifier(max_depth=25)),
    ("linear_svc",          LinearSVC(random_state=0)),
    ("gaussian",            GaussianNB()),
    ("multinomial",         MultinomialNB()),
    ("bernoulli",           BernoulliNB()),
    ("logistic_regression", LogisticRegression())
)

X_train = pd.read_csv("data/train_data.csv", sep=",", header=None)
X_validation = pd.read_csv("data/test_data.csv", sep=",", header=None)
y_train = pd.read_csv("data/train_labels.csv", sep=",", header=None)

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)

train, test, t_train, t_test = train_test_split(X_train, y_train, test_size=0.3)

for s, clf in classifiers:
    clf.fit(train, t_train.values.ravel())
    print("{:20}:".format(s), clf.score(test, t_test))


exit()

from sklearn.calibration import CalibratedClassifierCV

clf = CalibratedClassifierCV(LinearSVC(random_state=0))
clf.fit(X_train, y_train.values.ravel())
y_prediction = clf.predict(X_validation)

df = pd.DataFrame({'Sample_id':np.arange(1, 6545), 'Sample_label':y_prediction})
df.to_csv("results/accuracy_svc.csv", index=False)

prob = clf.predict_proba(X_validation)
df = pd.DataFrame(prob, columns=['Class_{}'.format(i+1) for i in range(10)])
df.insert(loc=0, column='Sample_id', value=np.arange(1, 6545))
df.to_csv("results/logloss_svc.csv", index=False)
