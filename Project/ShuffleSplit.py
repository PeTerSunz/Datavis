__author__ = 'Nattachai Chaiwiriya'
from sklearn.model_selection import ShuffleSplit
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
n_samples = iris.data.shape[0]
clf = svm.SVC(kernel='linear', C=1)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
print (cross_val_score(clf, iris.data, iris.target, cv=cv))


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(iris.data)
print(kf)
X = iris.data
y = iris.target
for train_index, test_index in kf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

