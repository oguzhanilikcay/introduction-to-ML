import matplotlib.pyplot as plt
import mglearn.plots
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

from sklearn.linear_model import LogisticRegression
logr0 = LogisticRegression()
logr0.fit(X_train, y_train)

score = logr0.score(X_test, y_test)
print("test score: {:.2f}".format(score))


#
from sklearn.datasets import load_iris
iris = load_iris()

logr = LogisticRegression(max_iter=200)

# cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logr, iris.data, iris.target, cv=5)
print("cross-validation scores: {}".format(scores))
print("average cross-validation score: {:.2f}".format(scores.mean()))


#
# stratified k-fold cross-validation
#

iris = load_iris()
print("Iris labels:\n{}".format(iris.target))

mglearn.plots.plot_stratified_cross_validation()
plt.show()

#
# more control over cross-validation
#

from sklearn.model_selection import KFold
cv = KFold(n_splits=3)  # !
print("cross-validation scores:\n{}".format(
    cross_val_score(logr, iris.data, iris.target, cv=cv)))

cv = KFold(n_splits=3, shuffle=True, random_state=0)
print("cross-validation scores:\n{}".format(
    cross_val_score(logr, iris.data, iris.target, cv=cv)))

#
# leave-one-out cross-validation
#

from sklearn.model_selection import LeaveOneOut
cv = LeaveOneOut()
scores = cross_val_score(logr, iris.data, iris.target, cv=cv)
print("number of iterations: ", len(scores))
print("mean accuracy: {:.2f}".format(scores.mean()))

#
# shuffle-split cross-validation
#

mglearn.plots.plot_shuffle_split()
plt.show()

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10)
scores = cross_val_score(logr, iris.data, iris.target, cv=cv)
print("cross-validation scores:\n{}".format(scores))

#
# group-k-fold
#

X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=3)
scores = cross_val_score(logr, X, y, groups=groups, cv=cv)
print("cross-validation scores:\n{}".format(scores))

mglearn.plots.plot_group_kfold()
plt.show()
