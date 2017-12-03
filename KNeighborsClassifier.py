from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv

X_train = pd.read_csv("data/train_data.csv", sep=",", header=None)
X_validation = pd.read_csv("data/test_data.csv", sep=",", header=None)
y_train = pd.read_csv("data/train_labels.csv", sep=",", header=None)

"""
X_harjoitus, X_testi, y_harjoitus, y_testi = train_test_split(X_train, y_train, train_size=0.9, test_size=0.1, random_state=12)
clf = KNeighborsClassifier()
clf.fit(X_harjoitus, y_harjoitus.values.ravel())
y_pred = clf.predict(X_testi)
print(accuracy_score(y_pred, y_testi))
print(y_pred)
"""

clf = KNeighborsClassifier()
clf.fit(X_train, y_train.values.ravel())
y_prediction = clf.predict(X_validation)
y_list = np.arange(6544)

with open('accuracy.csv', 'w', newline='') as csvfile:
    write = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    write.writerow(['Sample_id']+['Sample_label'])
    for i in range(1,6545):
        write.writerow([i] + [y_prediction[i-1]])
