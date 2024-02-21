import numpy as np
import matplotlib.pyplot as plt
import mglearn
import seaborn
import pandas as pd

# options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# dataset (type 2)
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=2, cluster_std=[5.0, 1.25], random_state=30)

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.25
)

from sklearn.svm import SVC
svm = SVC(C=0.1, gamma=1.0, kernel="linear", random_state=0)
svm.fit(X_train, y_train)

score_train = svm.score(X_train, y_train)
score_test = svm.score(X_test, y_test)

y_pred = svm.predict(X_test)

print("train score: {:.3f}\ntest score: {:.3f}".format(score_train, score_test))



#
plt.scatter(X[:, 0], X[:, 1], s=20, marker='o', c=y, edgecolor='k')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.show()

#
from sklearn.metrics import classification_report
c_rep = classification_report(y_test, y_pred)
print("\n< classification report >\n{}".format(c_rep))


#
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("< confusion matrix >\n{}".format(cm))

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print(TN, FP, FN, TP)


#
df = pd.DataFrame({'X0': X[:, 0], 'X1': X[:, 1], 'label': y})
print(df.head())



from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("accuracy_score: {:.2f}".format(score))



'''         plotting decision boundary          '''

def make_mashgrid(x, y, h=0.02):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)

    return out

fig, ax = plt.subplots()
xx, yy = make_mashgrid(X[:, 0], X[:, 1])

plot_contours(ax, svm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.6)
ax.set_ylabel('y label')
ax.set_xlabel('x label')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('decision surface of linear SVC')
ax.legend()

plt.show()
