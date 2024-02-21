import mglearn
import matplotlib.pyplot as plt
import numpy as np


# non-negative matrix factorization
mglearn.plots.plot_nmf_illustration()
plt.show()

#
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


mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
plt.show()
