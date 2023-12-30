import mglearn
import matplotlib.pyplot as plt
import numpy as np

mglearn.plots.plot_kmeans_algorithm()
plt.show()

mglearn.plots.plot_kmeans_boundaries()
plt.show()

#
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=1)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(X)

print("cluster membership:\n{}".format(km.labels_))
print(km.predict(X))


mglearn.discrete_scatter(X[:, 0], X[:, 1], km.labels_, markers=['o'])
mglearn.discrete_scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                         [0, 1, 2], markers=['^'], markeredgewidth=2)
plt.show()



#
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# using two cluster centers:
kmeans_2c = KMeans(n_clusters=2)
kmeans_2c.fit(X)
assignments = kmeans_2c.labels_

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

# using five cluster centers:
kmeans_5c = KMeans(n_clusters=5)
kmeans_5c.fit(X)
assignments = kmeans_5c.labels_

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

plt.show()


# cluster assignments found by k-means when clusters have different densities
X_varied, y_varied = make_blobs(n_samples=200,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=170)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_varied)
y_pred = kmeans.predict(X_varied)

mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
plt.legend(['cluster 0', 'cluster 1', 'cluster 2'], loc='best')
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.show()


# k-means fails to indentify nonspherical clusters sample
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=[0, 1, 2], s=150, linewidth=2, cmap=mglearn.cm3)

plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.show()


# k-means fails to identify clusters with complex shapes
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=['r', 'g'], s=120, linewidth=2)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.show()



#
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=10, random_state=0)
km.fit(X)
y_pred = km.predict(X)


plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=40, cmap='Paired', alpha=0.9)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=150, marker='^', c=range(km.n_clusters), linewidth=2)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.show()

print("cluster membership:\n{}".format(y_pred))


distance_features = km.transform(X)
print("distance feature shape: {}".format(distance_features.shape))
print("destance features:\n{}".format(distance_features))
