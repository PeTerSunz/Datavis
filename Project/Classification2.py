__author__ = 'Nattachai Chaiwiriya'

import pandas as pd
import csv
import numpy as np
TRAIN_FILE_LOCATION1 = 'D:\\Data_vis\\nursery\\nursery\\nursery-5-2tra.csv'
TEST_FILE_LOCATION1 = 'D:\\Data_vis\\nursery\\nursery\\nursery-5-2tst.csv'


def ConvertCategoricalToNumericColumn(df):
    g = df.columns.to_series().groupby(df.dtypes).groups
    check = None
    for k, v in g.items():
        if k.name == "object":
            check = True
            for index in v:
                df[index] = df[index].astype('category')
    if check:
        cat_columns = df.select_dtypes(include=['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

train_data = pd.read_csv(TRAIN_FILE_LOCATION1)
train_df = pd.DataFrame(data = train_data)
test_data = pd.read_csv(TEST_FILE_LOCATION1)
test_df = pd.DataFrame(data = test_data)

print("_________________________________train")
print (train_df)
print("_________________________________test")
print (test_df)


train_data = pd.read_csv(TRAIN_FILE_LOCATION1)
train_df = pd.DataFrame(data = train_data)
test_data = pd.read_csv(TEST_FILE_LOCATION1)
test_df = pd.DataFrame(data = test_data)


#test_df2 = test_df.copy(deep=True)
##

train_df = ConvertCategoricalToNumericColumn(train_df)
test_df = ConvertCategoricalToNumericColumn(test_df)

print("_________________________________train")
print (train_df)
print("_________________________________test")
print (test_df)


# Decision Tree
X_train = train_df.iloc[:,:-1].values
y_train = train_df.iloc[:,-1].values
X_test = test_df.iloc[:,:-1].values
y_test = test_df.iloc[:,-1].values

# X_train2 = train_df.iloc[:,:-1].values

from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)


print("_______________Accuracy DT________________")
print (clf.score(X_test, y_test) * 100)

'''
# # Support Vector Machines
# from sklearn import svm
#
# clf = svm.SVC(gamma='scale')
# clf.fit(X_train, y_train)
#
# print("_______________Accuracy________________")
# print (clf.score(X_test, y_test) * 100)

# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

print("_______________Accuracy KN________________")
print (neigh.score(X_test,y_test) *100)


# Neural network
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)

print("_______________Accuracy NN________________")
print (clf.score(X_test,y_test) *100)
'''

# Predict Model
prediction = clf.predict(X_test)
print("_______________prediction ________________")
print (prediction)


# merge X_test with prediction values from classifier model
prediction = np.array([prediction])
prediction = np.concatenate((X_test, prediction.T), axis=1)
print("_______________prediction 2________________")
print(prediction)

OUTPUT_FILE_LOCATION = 'D:\\Data_vis\\nursery\\nursery\\test.csv'

# Write Column Header into first line of CSV file

f = open(OUTPUT_FILE_LOCATION, 'w')
colno = 0
for item in test_df.columns.values:
    colno += 1
    if colno < len(test_df.columns.values):
        f.write("%s," % item)
    else:
        f.write("%s" % item)
f.write("\n")
f.close()


f = open(OUTPUT_FILE_LOCATION, 'a')
writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
for row in prediction:
    f.write(','.join(map(str, row)) )
    f.write("\n")
f.close()