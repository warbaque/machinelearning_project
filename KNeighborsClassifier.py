from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X_train = pd.read_csv("data/train_data.csv", sep=",", header=None)
X_validation = pd.read_csv("data/test_data.csv", sep=",", header=None)
y_train = pd.read_csv("data/train_labels.csv", sep=",", header=None)

'''
train, test, t_train, t_test = train_test_split(X_train, y_train, test_size=0.33)
clf = KNeighborsClassifier()
clf.fit(train, t_train.values.ravel())
print(clf.score(test, t_test))
exit()
'''

clf = KNeighborsClassifier()
clf.fit(X_train, y_train.values.ravel())
y_prediction = clf.predict(X_validation)

df = pd.DataFrame({'Sample_id':np.arange(1, 6545), 'Sample_label':y_prediction})
df.to_csv("results/accuracy_kn.csv", index=False)
