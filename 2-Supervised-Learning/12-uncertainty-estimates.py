import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import make_circles
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# rename the classes for illustration purposes
y_named = np.array(['blue', 'red'])[y]

# we can call train_test_split with arbitrarily many arrays; all will be split in a consistent manner
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(
    X, y_named, y, random_state=0
)


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train_named)

# The Decision Function
''' if the absolute value of the score is large, the model's confidence in classification increases'''

print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(gbc.decision_function(X_test).shape))

print("Decision function:\n{}".format(gbc.decision_function(X_test[:6])))

print("Thresholded decision function:\n{}".format(gbc.decision_function(X_test) > 0))
print("Predictions: {}".format(gbc.predict(X_test)))


# make the boolean True/False into 0 and 1
greater_zero = (gbc.decision_function(X_test) > 0).astype(int)
# use 0 and 1 as indices into 'classes_'
pred = gbc.classes_[greater_zero]
# pred is the same as the output of gbc.predict
print("pred is equal to predictions: {}".format(np.all(pred == gbc.predict(X_test))))


decision_function = gbc.decision_function(X_test)
print("Decision function minimum: {:.2f} maximum: {:.2f}".format(
    np.min(decision_function), np.max(decision_function)))


# plot decision boundary and decision function
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

mglearn.tools.plot_2d_separator(gbc, X, ax=axes[0], alpha=0.4, fill=True, cm=mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbc, X, ax=axes[1], alpha=0.4, cm=mglearn.ReBl)

for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, ax=ax, markers='^')
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax, markers='o')
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    axes[0].set_title('Decision Boundary')
    axes[1].set_title("Decision Function")

cbar = plt.colorbar(score_image, ax=axes.tolist())
axes[0].legend(['test class 0', 'test class 1', 'train class 0', 'train class 1'],
               ncol=4, loc=(0.1, 1.1))

plt.show()


print("Shape of probabilities: {}".format(gbc.predict_proba(X_test).shape))
print("Predicted probabilities:\n{}".format(gbc.predict_proba(X_test[:5])))


# plot decision boundary and predicted probabilities
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbc, X, ax=axes[0], alpha=0.4, fill=True, cm=mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbc, X, ax=axes[1], alpha=0.5, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    axes[0].set_title('Decision Boundary')
    axes[1].set_title('Predicted Probabilities')

cbar = plt.colorbar(score_image, ax=axes.tolist())
axes[0].legend(['test class 0', 'test class 1', 'train class 0', 'train class 1'],
               ncol=4, loc=(0.1, 1.1))

plt.show()
