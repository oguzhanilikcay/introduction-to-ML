import matplotlib.pyplot as plt
import mglearn

mglearn.plots.plot_agglomerative_algorithm()
plt.show()

#
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=1)

from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.show()


mglearn.plots.plot_agglomerative()
plt.show()


# dendrogram
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)
linkage_array = ward(X)
''' ward(), kume icindeki veri noktalarinin varyansini dusuk tutarak homojen kumeleme yapmayi saglar'''


# plot the dendrogram for the linkage_array containing the distances between clusters
dendrogram(linkage_array)

# mark the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel('sample index')
plt.ylabel('cluster distance')

plt.show()


#
X, y = make_blobs(random_state=0, n_samples=12)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)

print("cluster membershpis:\n{}".format(clusters))  # -1 means noise


mglearn.plots.plot_dbscan()
plt.show()

#
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_scaled = sc.transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, s=50)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

plt.show()

'''
> increasing 'eps' means that more points will be included in a cluster. this makes clusters grow,
but might also lead to multiple cluster joining into one.
> increasing 'min_samples' means that fewer points will be core points, and more points will
be labeleed as noise.
'''
