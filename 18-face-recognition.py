import numpy as np
import mglearn

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

print(image_shape)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

print("people.images.shape: {}".format(people.images.shape))
print("number of classes: {}".format(len(people.target_names)))

print("people.images.shape: {}".format(people.images.shape))
print("number of classes: {}".format(len(people.target_names)))

# count how often each target appears
counts = np.bincount(people.target)

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()


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

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("\n> test set score of n=1: {:.2f}".format(knn.score(X_test, y_test)))


mglearn.plots.plot_pca_whitening()
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca_shape: {}".format(X_train_pca.shape))


knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(X_train_pca, y_train)

print("> test set accuracy: {:.2f}".format(knn2.score(X_test_pca, y_test)))












