__author__ = 'Nattachai Chaiwiriya'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print (iris.data.shape, iris.target.shape)

# 40% of the data for testing classifier:
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Train a linear support vector machine
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print (clf.score(X_test, y_test))



from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print (scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
