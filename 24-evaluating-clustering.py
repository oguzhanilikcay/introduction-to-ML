import matplotlib.pyplot as plt
import numpy as np


# ARI
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_scaled = sc.transform(X)



from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
algorithms = [
    KMeans(n_clusters=2),
    AgglomerativeClustering(n_clusters=2),
    DBSCAN()
]


random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))


fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, s=60)
axes[0].set_title("Random assingment - ARI: {:.2f}".format(
    adjusted_rand_score(y, random_clusters)))

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, s=60)
    ax.set_title("{} - ARI {:.2f}".format(
        algorithm.__class__.__name__, adjusted_rand_score(y, clusters)))

plt.show()



# a common mistake sample
from sklearn.metrics import accuracy_score

clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]
print("accuracy_score: {:.2f}".format(accuracy_score(clusters1, clusters2)))  # equals 0.0
print("ARI: {}".format(adjusted_rand_score(clusters1, clusters2)))  # equals 1.0

'''
> a common mistake when evalutaing clustering in this way is to use 'accuracy_score'
instead of 'adjusted_rand_score, normalized_mutual_info_score, or other clustering metrics'.
'''


# silhouette coefficient
from sklearn.metrics import silhouette_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

sc1 = StandardScaler()
sc1.fit(X)
X_scaled = sc1.transform(X)

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))


fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, s=60)
axes[0].set_title("Random assignment: {:.2f}".format(
    silhouette_score(X_scaled, random_clusters)))

algorithms = [
    KMeans(n_clusters=2),
    AgglomerativeClustering(n_clusters=2),
    DBSCAN()
]

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, s=60)
    ax.set_title("{} : {:.2f}".format(
        algorithm.__class__.__name__, silhouette_score(X_scaled, clusters)))

plt.show()
