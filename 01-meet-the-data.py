import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas.plotting as pdp
import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()

print("keys of iris dataset: \n{}".format(iris.keys()))
print(iris['DESCR'][:193] + '\n...')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0, test_size=0.25
)

iris_df = pd.DataFrame(X_train, columns=iris.feature_names)


# create a scatter matrix from the dataframe
grr = pdp.scatter_matrix(iris_df, c=y_train, figsize=(20, 20),
                         marker='o',hist_kwds={'bins': 20}, s=13,
                         alpha=0.85, cmap=mglearn.cm3)

plt.show()


# < / >
X, y = mglearn.datasets.make_forge()
print('>X.shape: {}'.format(X.shape))

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Class 0', 'Class 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')

plt.show()


#
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('feature')
plt.ylabel('target')
plt.show()


#
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("> Cancer.keys(): \n{}".format(cancer.keys()))
print("> Shape of cancer data: {}".format(cancer.data.shape))
print("> Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("> Feature names:\n{}".format(cancer.feature_names))
