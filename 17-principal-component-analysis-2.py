import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(cancer.data)
X_scaled = sc.transform(cancer.data)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)

print("original shape: {}".format(str(X_scaled.shape)))
print("reduced shape: {}".format(str(X_pca.shape)))


plt.figure(figsize=(8, 8))

import mglearn
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)

plt.legend(cancer.target_names, loc='best')
plt.xlabel('first principal component')
plt.ylabel('second principal componant')
plt.gca().set_aspect('equal')

plt.show()

print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['first component', 'second component'])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation=60, ha='left')
plt.xlabel('feature')
plt.ylabel('principal component')
plt.title('heat map of the first two principle components')

plt.show()
