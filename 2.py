import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split

#       < Linear Models >
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=60
)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print("> lr.coef_: {}".format(lr.coef_))
print("> lr.intercept_: {}".format(lr.intercept_))


print("> Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("> Test set score: {:.2f}".format(lr.score(X_test, y_test)))

'''
An R-square of around 0.66 is not very good, but we can see that the scores on the training
and test sets are very close together. This means we are likely underfitting, not over‐
fitting. For this one-dimensional dataset, there is little danger of overfitting, as the
model is very simple (or restricted). However, with higher-dimensional datasets
(meaning datasets with a large number of features), linear models become more pow‐
erful, and there is a higher chance of overfitting. Let’s take a look at how LinearRe
gression performs on a more complex dataset, like the Boston Housing dataset.
Remember that this dataset has 506 samples and 105 derived features. First, we load
the dataset and split it into a training and a test set. Then we build the linear regres‐
sion model as before: 
'''


# boston house - LR
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)
lr = LinearRegression()
lr.fit(X_train, y_train)

print(">> Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print(">> Test set score: {:.2f}".format(lr.score(X_test, y_test)))


# Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)

print("> Training set score (ridge): {:.2f}".format(ridge.score(X_train, y_train)))
print("> Test set score (ridge): {:.2f}".format(ridge.score(X_test, y_test)))

'''
The Ridge model makes a trade-off between the simplicity of the model (near-zero
coefficients) and its performance on the training set
'''

alphas = [0.01, 0.1, 0.5, 1, 5, 10, 12]
for alpha in alphas:
    ridge_v2 = Ridge(alpha=alpha)
    ridge_v2.fit(X_train, y_train)
    print("--" * 5)
    print("alpha value = {}".format(alpha))
    print("train score: {:.2f}".format(ridge_v2.score(X_train, y_train)))
    print("test score: {:.2f}".format(ridge_v2.score(X_test, y_test)))


ridge1 = Ridge(alpha=0.1)
ridge1.fit(X_train, y_train)

ridge2 = Ridge(alpha=1)
ridge2.fit(X_train, y_train)

ridge3 = Ridge(alpha=10)
ridge3.fit(X_train, y_train)


plt.plot(ridge1.coef_, 's', label='alpha=0.1')
plt.plot(ridge2.coef_, '^', label='alpha=1.0')
plt.plot(ridge3.coef_, 'v', label='alpha=10.0')

plt.plot(lr.coef_, 'o', label='Linear Regression')
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()



#   Lasso
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)

print("train score (lasso): {:.2f}".format(lasso.score(X_train, y_train)))
print("test score (lasso): {:.2f}".format(lasso.score(X_test, y_test)))
print("number of features used: {}".format(np.sum(lasso.coef_ != 0)))


# Lasso with params
lasso1 = Lasso(alpha=0.01, max_iter=100000)
lasso1.fit(X_train, y_train)

print("train score (lasso1): {:.2f}".format(lasso1.score(X_train, y_train)))
print("test score (lasso1): {:.2f}".format(lasso1.score(X_test, y_test)))
print("number of features used: {:.2f}".format(np.sum(lasso1.coef_ != 0)))

''' A lower alpha allowed us to fit more complex model '''


# comparison
plt.plot(lasso.coef_, 's', label='Lasso alpha=1')
plt.plot(lasso1.coef_, 'o', label='Lasso alpha=0.01')
plt.plot(ridge1.coef_, '^', label='Ridge alpha= 0.1')

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')

plt.show()



'''
     y = w[0] * x[0] + ... + w[p] * x[p] + b
        > p is the number of features
        > w, b are parameters of the modeol that are learned
        > w also called coefficient. it is stored in the 'coef_' attribute
        > b also called intercept. it is stored in the 'intercept_' attribute
        > y is the prediction the model makes

> we want the magnitude of coefficients to be as small as possible.
> regularization means explicitly restricting a model to avoid overfitting.

'''