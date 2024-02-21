import numpy as np
import matplotlib.pyplot as plt
import mglearn
import seaborn

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


'''                 study metrics for classification            '''

# 0 - analyzing the data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=360, centers=3, random_state=9, cluster_std=2.15)

print("first 10 indexes\n{}".format(X[:10]))


print("X shape: {}".format(X.shape))
print("values of first index:\n{}".format(X[0]))

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("dataset")

plt.show()


print("first columns (min,max) values: ({:.2f}, {:.2f})".format(np.min(X[:, 0]), np.max(X[:, 0])))
print("second columns (min,max) values: ({:.2f}, {:.2f})".format(np.min(X[:, 1]), np.max(X[:, 1])))

print("std_dev of the first col: {:.2f}".format(np.std(X[:, 0])))
print("std_dev of the second col: {:.2f}".format(np.std(X[:, 1])))

print("unique values of target (y): {}".format(np.unique(y)))



#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)


# 1 - Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1.1
logr = LogisticRegression()
logr.fit(X_train, y_train)

# 1.2
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# 1.3
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# 1.4
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

models = [logr, dtc, rfc, svc]

# plotting the boundaries
for model in models:
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    mglearn.plots.plot_2d_classification(model, X, fill=False, alpha=0.3)

    plt.title("{}".format(model))
    plt.show()
    pass

# predictions
y_pred_logr = logr.predict(X_test)
y_pred_dtc = dtc.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
y_pred_svc = svc.predict(X_test)


#  - Classification Metrics -
'''
some metrics are essentially defined for binary classifcation (f1_score, roc_auc_score)
'''
# 1 - confusion_matrix
from sklearn.metrics import confusion_matrix

cm_logr = confusion_matrix(y_test, y_pred_logr)
print("confusion matrix (logr):\n{}".format(cm_logr))

cm_dtc = confusion_matrix(y_test, y_pred_dtc)
print("confusion matrix (dtc):\n{}".format(cm_dtc))

cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print("confusion matrix (rfc):\n{}".format(cm_rfc))

cm_svc = confusion_matrix(y_test, y_pred_svc)
print("confusion matrix (svc):\n{}".format(cm_svc))


seaborn.heatmap(
    cm_logr,
    cmap=plt.cm.gray_r,
    annot=True
)

plt.xlabel("predicted label")
plt.ylabel("true label")
plt.title("confusion matrix (logr)")

plt.show()


# - 2 - cross_val_score


#
from sklearn.model_selection import cross_val_score

svc = SVC(kernel='linear', C=1, random_state=0)

scores = cross_val_score(svc, X, y, cv=5)
print("cv scores: {}".format(scores))
print("average cv scores : {:.2f}".format(scores.mean()))
print("standard deviation of scores: {:.2f}".format(scores.std()))

svc.fit(X_train, y_train)
print("train score: {:.2f}".format(svc.score(X_train, y_train)))
print("test score: {:.2f}".format(svc.score(X_test, y_test)))


# ShuffleSplit
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.5, train_size=0.5, random_state=1)
scores = cross_val_score(svc, X, y, cv=cv)

print("cv scores: {}".format(scores))
print("average cv scores : {:.2f}".format(scores.mean()))
print("standard deviation of scores: {:.2f}".format(scores.std()))



# cross_validate
from sklearn.model_selection import cross_validate

scores = cross_validate(svc, X, y)
print(scores.keys())
print(scores['test_score'])
