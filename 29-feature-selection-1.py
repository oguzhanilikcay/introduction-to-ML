import matplotlib.pyplot as plt
import numpy as np


#
# Univariate Statistics
#


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

X_w_noise = np.hstack([cancer.data, noise])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=0.5
)


from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape {}".format(X_train_selected.shape))

mask = select.get_support()
print(mask)

plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')
plt.title("SelectPercentile")

plt.show()

# compare the performance on all features against the performance using only the selected features:
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=0.05)

logr.fit(X_train, y_train)
score = logr.score(X_test, y_test)
print("score with all features: {:.3f}".format(score))

logr.fit(X_train_selected, y_train)
score_selected = logr.score(X_test_selected, y_test)
print("score with only selected features: {:.3f}".format(score_selected))


#
# Model-Based Feature Selection
#


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
X_test_l1 = select.transform(X_test)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample index')
plt.title('SelectFromModel')

plt.show()


logr = LogisticRegression(C=0.05)
logr.fit(X_train_l1, y_train)
score = logr.score(X_test_l1, y_test)
print("test score: {:.3f}".format(score))


#
# Iterative Feature Selection
#


from sklearn.feature_selection import RFE
select = RFE(
    RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=40
)

select.fit(X_train, y_train)

mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample index')
plt.title('RFE')

plt.show()


X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

logr = LogisticRegression(C=0.05)
logr.fit(X_train_rfe, y_train)
score = logr.score(X_test_rfe, y_test)
print("test score: {:.3f}".format(score))

print("rfe.score: {:.3f}".format(select.score(X_test, y_test)))
