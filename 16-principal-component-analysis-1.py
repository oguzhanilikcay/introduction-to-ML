import mglearn
import matplotlib.pyplot as plt
import numpy as np


mglearn.plots.plot_pca_illustration()
plt.show()

#
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

malignant = cancer.data[cancer.target == 1]
benign = cancer.data[cancer.target == 0]

fig, axes = plt.subplots(15, 2, figsize=(10, 20))
ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color='red', alpha=0.5)
    ax[i].hist(benign[:, i], bins=bins, color='blue', alpha=0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel('Feature magnitude')
ax[0].set_ylabel('Frequency')
ax[0].legend(['malignant', 'benign'], loc='best')
fig.tight_layout()

plt.show()


#
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
