import pandas as pd
import csv
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

FILE_LOCATION = 'D:\\Data_vis\\quake.csv'
OUTPUT_LOCATION = 'D:\\Data_vis\\quake_cluster.csv'
data = pd.read_csv(FILE_LOCATION)
df = pd.DataFrame(data = data)
X = df.iloc[:,:].values
print (X)
print (len(X))


Clustering = KMeans(n_clusters=2, random_state=0).fit(X)
print (Clustering.labels_)
print (Clustering.cluster_centers_)

# Count Number of points in each cluster
counter = Counter(Clustering.labels_)
result = [(key, counter[key]) for key in counter]
print (result)


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

Clustering = MeanShift(bandwidth=bandwidth,bin_seeding=True).fit(X)

labels = Clustering.labels_
cluster_centers = Clustering.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

Clustering = AgglomerativeClustering(n_clusters = 5,linkage='complete').fit(X)
print (Clustering.labels_)
print (Clustering)


# Count Number of points in each cluster
counter = Counter(Clustering.labels_)
result = [(key, counter[key]) for key in counter]
print (result)

Clustering = DBSCAN(eps=30, min_samples=2).fit(X)
print (Clustering.labels_)
print (Clustering)

labels = Clustering.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Add Cluster ID column
df['cluster'] = labels.tolist()
df.to_csv(OUTPUT_LOCATION, sep=',', encoding='utf-8')