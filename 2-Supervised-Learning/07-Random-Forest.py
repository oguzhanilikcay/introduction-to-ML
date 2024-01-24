import numpy as np
import matplotlib.pyplot as plt
import mglearn


#   dataset - make_moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5, random_state=2)
rfc.fit(X_train, y_train)

print("rfc training set accuray: {:.2f}".format(rfc.score(X_train, y_train)))
print("rfc test set accuracy {:.2f}".format(rfc.score(X_test, y_test)))


# visualize the decision boundaries learned by each tree
fig, axes = plt.subplots(2, 3, figsize=(30, 15))

for i, (ax, tree) in enumerate(zip(axes.ravel(), rfc.estimators_)):
    ax.set_title("Tree {}:".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(rfc, X_train, fill=True, ax=axes[-1, -1], alpha=0.4)
axes[-1, -1].set_title('Random Forest')

mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

plt.show()


#   dataset - cancer
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

rfc2 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc2.fit(X_train, y_train)

print("rfc2 training set accuray: {:.2f}".format(rfc2.score(X_train, y_train)))
print("rfc2 test set accuracy {:.2f}".format(rfc2.score(X_test, y_test)))


# visualize the coefficients
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title('Feature importance table')

    plt.show()


plot_feature_importances_cancer(rfc2)

'''
> n_estimators : the number of trees in the forest
> important parameters : {n_estimators, max_features, max_depth}
> "max_features" determines how random each tree is. smaller value reduces overfitting
> a good rule for max_features =
                               = sqrt(n_features) , for classification
                               = log2(n_features) , for regression 

> "max_features" or "max_leaf_nodes" might sometimes improve performance. it can also drastically
reduce space and time requirements for training and prediction
'''
