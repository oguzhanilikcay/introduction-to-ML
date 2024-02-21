import matplotlib.pyplot as plt
import mglearn
import numpy as np


X, y = mglearn.datasets.make_forge()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("> Test set predictions:\n{}".format(y_pred))

score = clf.score(X_test, y_test)
print("> Test set accuracy:\n{:.2f}".format(score))


# analyzing KNeighborsClassifier
fig, axes = plt.subplots(1, 3, figsize=(5, 5))

for n_neighbors, ax in zip([1, 3, 5], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)

    ax.set_title("{} neighbors(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

axes[0].legend(loc=3)

plt.show()


#
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66
)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(np.round(clf.score(X_train, y_train), 3))
    test_accuracy.append(np.round(clf.score(X_test, y_test), 3))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs n_neighbors')
plt.legend()

plt.show()

print("training accuracy: {}".format(training_accuracy))
print("test accuracy: {}".format(test_accuracy))


# KNN regression
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("> Test set prediction:\n{}".format(y_pred))

score = reg.score(X_test, y_test)
print("> Test set R-square: {:.2f}".format(score))


# analyzing KNeighborsRegressor
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)  # create 1000 data points, evenly spaces between -3 and 3

for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)

    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "n={}\n train: {:.2f} test: {:.2f}".format(
            n_neighbors,
            reg.score(X_train, y_train),
            reg.score(X_test, y_test)
        )
    )
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')

axes[0].legend(['Model prediction', 'Training data/target', 'Test data/target'], loc='best')

plt.show()
