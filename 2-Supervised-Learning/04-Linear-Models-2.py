import matplotlib.pyplot as plt
import mglearn


# < LogisticRegression and Linear Support Vector Machines >
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, ax=ax, alpha=0.7, eps=0.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[0].legend()

plt.show()


mglearn.plots.plot_linear_svc_regularization()
plt.show()


#
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

logr = LogisticRegression(max_iter=10000)
logr.fit(X_train, y_train)

print("train score (logr): {:.5f}".format(logr.score(X_train, y_train)))
print("test score (logr): {:.5f}".format(logr.score(X_test, y_test)))


C_values = [0.01, 1, 100]

for c in C_values:
    logreg = LogisticRegression(max_iter=10000, C=c)
    logreg.fit(X_train, y_train)
    print("-" * 35)
    print("C:{}\ntrain score: {:.3f}\ntest score: {:.3f}".format(
        c,
        logreg.score(X_train, y_train),
        logreg.score(X_test, y_test)))

#
c_values = [0.001, 1, 100]
logr_models = []

for c in c_values:
    logr_x = LogisticRegression(max_iter=10000, C=c)
    logr_x.fit(X_train, y_train)
    logr_models.append(logr_x)

for i, model in enumerate(logr_models):
    label = "C={}".format(c_values[i])
    plt.plot(model.coef_.T, 'o', label=label, markersize=8)

plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim([-5, 5])
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.legend()

plt.show()

#
logr_l1_models = []
c_values = [0.001, 1, 100]

for c in c_values:
    logr_l1 = LogisticRegression(C=c, penalty='l1', solver='liblinear', max_iter=10000)
    logr_l1.fit(X_train, y_train)
    logr_l1_models.append(logr_l1)

    print("**" * 20)
    print("train accuracy (l1 logr) with C={}: {:.2f}".format(
        c, logr_l1.score(X_train, y_train)))
    print("test accuracy (l1 logr) with C={}: {:.2f}".format(
        c, logr_l1.score(X_test, y_test)))


for i, model in enumerate(logr_l1_models):
    label = "C={}".format(c_values[i])
    plt.plot(model.coef_.T, 'o', label=label, markersize=6)

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.title("penalty=l1")

plt.show()

''' the penalty parameters work with different solvers'''
