import numpy as np
import matplotlib.pyplot as plt


# Gradient Boosted Regression Trees (gradient boosting machines)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)


# compare parameters
from sklearn.ensemble import GradientBoostingClassifier

# -1-
gbc1 = GradientBoostingClassifier(random_state=0)
gbc1.fit(X_train, y_train)

print("train set accuracy(non parameter): {:.3f}".format(gbc1.score(X_train, y_train)))
print("test set accuracy(non parameter): {:.3f}".format(gbc1.score(X_test, y_test)))

# -2-
gbc2 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbc2.fit(X_train, y_train)

print("train set accuracy(max_depth=1): {:.3f}".format(gbc2.score(X_train, y_train)))
print("test set accuracy(max_depth=1): {:.3f}".format(gbc2.score(X_test, y_test)))

# -3-
gbc3 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbc3.fit(X_train, y_train)

print("train set accuracy(learning_rate=0.01): {:.3f}".format(gbc3.score(X_train, y_train)))
print("test set accuracy(learning_rate=0.01) : {:.3f}".format(gbc3.score(X_test, y_test)))


# visualize the coefficients
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title('> {}'.format(model))

    plt.show()


plot_feature_importances_cancer(gbc1)
plot_feature_importances_cancer(gbc2)
plot_feature_importances_cancer(gbc3)


'''
> gradient boosted decision trees are among the most powerful and widely used models for supervised learning. 
> they require careful tuning of the parameters and may take a long time to train.
> they work well without scaling.
> it may not work well on high-dimensional sparse data.
> main parameters : {n_estimators, learning_rate, max_depth, max_leaf_nodes}
> "n_estimators", "learning_rate" they control the degree to which each tree is allowed to correct the
mistakes of the previous trees.
> increasing "n_estimators" in gradient boosting leads to a more comples model, which may lead to overfitting.
'''
