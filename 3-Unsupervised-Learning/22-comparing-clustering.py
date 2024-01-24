import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

image_shape = people.images[0].shape
counts = np.bincount(people.target)

mask = np.zeros(people.target.shape, dtype=np.bool_)

for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0
)


#
from sklearn.decomposition import NMF
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)

from sklearn.decomposition import PCA
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=100, random_state=0)
km.fit(X_train)


X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_km = km.cluster_centers_[km.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)


fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('Extracted Components')

for ax, comp_km, comp_pca, comp_nmf in zip(
    axes.T, km.cluster_centers_, pca.components_, nmf.components_
):
    ax[0].imshow(comp_km.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))
    axes[0, 0].set_ylabel('k-means')
    axes[1, 0].set_ylabel('pca')
    axes[2, 0].set_ylabel('nmf')

fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(8, 8))
fig.suptitle('Reconstructions')

for ax, orig, rec_km, rec_pca, rec_nmf in zip(
    axes.T, X_test, X_reconstructed_km, X_reconstructed_pca, X_reconstructed_nmf
):
    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_km.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))
    ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel('original')
axes[1, 0].set_ylabel('k-means')
axes[2, 0].set_ylabel('pca')
axes[3, 0].set_ylabel('nmf')

plt.show()
