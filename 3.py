import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split


#
X, y = mglearn.datasets.make_forge()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    classifier = model.fit(X, y)
    mglearn.plots.plot_2d_separator(classifier, X, fill=False, ax=ax, alpha=0.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(classifier.__class__.__name__))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[0].legend()

plt.show()



#
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

logr = LogisticRegression(max_iter=10000)
logr.fit(X_train, y_train)

print("train score (logr): {:.6f}".format(logr.score(X_train, y_train)))
print("test score (logr): {:.6f}".format(logr.score(X_test, y_test)))


C_values = [0.01, 1, 100]

for c in C_values:
    logreg = LogisticRegression(max_iter=10000, C=c)
    logreg.fit(X_train, y_train)
    print("-" * 10)
    print("C:{}, train score: {:.3f}, test score: {:.3f}".format(
        c,
        logreg.score(X_train, y_train),
        logreg.score(X_test, y_test)
    ))


