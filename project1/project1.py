import pandas as pd
import csv
import numpy as np

# paipayon verion
# Convert function
def ConvertDB(df):
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


# Convert train
FILE_LOCATION = 'C:\\Users\\Paipayon\\PycharmProjects\\project\\venv\\bank3.csv'
train_data = pd.read_csv(FILE_LOCATION)
train_df = pd.DataFrame(data=train_data)
train_df = ConvertDB(train_df)
# print (train_df)

# Convert test
FILE_LOCATION1 = 'C:\\Users\\Paipayon\\PycharmProjects\\project\\venv\\bank1.csv'
test_data = pd.read_csv(FILE_LOCATION1)
test_df = pd.DataFrame(data=test_data)
test_df = ConvertDB(train_df)
# print (test_df)

# plots
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="darkgrid")

bank = pd.read_csv('C:\\Users\\Paipayon\\PycharmProjects\\project\\venv\\bank.csv');
# bank = train_df
#line
sns.relplot(x="job", y="age", hue="default", kind="line", data=bank);
# # Point Plots
sns.catplot(x="age", y="marital", hue="job", data=bank);
# # bar
sns.catplot(x="duration", y='job', hue="marital", kind="bar", data=bank);
#box
sns.catplot(x="age", y="marital", hue="housing",  kind="box", data=bank);

#test
plt.show()

# Train
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print clf.score(X_test, y_test)

# K-NN
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, y_train)
print neigh.score(X_test, y_test)

# Clustering
import pandas as pd
import csv
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth


df = train_df
# df = pd.DataFrame(data=train_df)
X = df.iloc[:, :].values



# Kmean

Clustering = KMeans(n_clusters=3, random_state=0).fit(X)
print (Clustering.labels_)
# print (Clustering.cluster_centers_)


# Mean Shift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
Clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
labels = Clustering.labels_
cluster_centers = Clustering.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

