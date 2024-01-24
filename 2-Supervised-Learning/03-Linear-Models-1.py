import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_breast_cancer

#       < Linear Models >
X, y = mglearn.datasets.make_wave(n_samples=60)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

# ordinary least squares (OLS)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print("> lr.coef_: {}".format(lr.coef_))
print("> lr.intercept_: {:.2f}".format(lr.intercept_))

print("> Train score: {:.2f}".format(lr.score(X_train, y_train)))
print("> Test score: {:.2f}".format(lr.score(X_test, y_test)))

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

#
# Compare the linear models (OLS, Ridge, Lasso) - for dataset : extended_boston
#
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# Linear Regression *
lr0 = LinearRegression()
lr0.fit(X_train, y_train)

print(">> train score (lr): {:.2f}".format(lr0.score(X_train, y_train)))
print(">> test score (lr): {:.2f}".format(lr0.score(X_test, y_test)))


# Ridge Regression *
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)

print("> Train score (ridge): {:.2f}".format(ridge.score(X_train, y_train)))
print("> Test score (ridge): {:.2f}".format(ridge.score(X_test, y_test)))

''' The Ridge model makes a trade-off between the simplicity of the model (near-zero
coefficients) and its performance on the training set '''

# compare ridge models with different alpha values
print("-"*10, "Ridge Models with different alpha values", "-"*10)

alphas = [0.1, 1, 10]
ridge_models = []

for alpha in alphas:
    ridge_x = Ridge(alpha=alpha)
    ridge_x.fit(X_train, y_train)
    print("alpha value = {}".format(alpha))
    print("train score: {:.2f}".format(ridge_x.score(X_train, y_train)))
    print("test score: {:.2f}".format(ridge_x.score(X_test, y_test)))
    print("---" * 10)
    ridge_models.append(ridge_x)

for i, model in enumerate(ridge_models):
    label = "Ridge alpha = {}".format(alphas[i])
    plt.plot(model.coef_, 'o', label=label)


plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
plt.hlines(0, 0, len(ridge_models[0].coef_))
plt.ylim(-2, 2)
plt.title('Ridge Regression Coefficients for Different Alpha Values')
plt.legend()

plt.show()


# Lasso *
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)

print("> Train score (lasso): {:.2f}".format(lasso.score(X_train, y_train)))
print("> Test score (lasso): {:.2f}".format(lasso.score(X_test, y_test)))
print("> number of features used: {}".format(np.sum(lasso.coef_ != 0)))


# compare lasso models with different alpha values
print("-"*10, "Lasso Models with different alpha values", "-"*10)

alphas = [0.001, 0.01, 0.1, 1]
lasso_models = []

for alpha in alphas:
    lasso_x = Lasso(alpha=alpha, max_iter=10000)
    lasso_x.fit(X_train, y_train)
    print("alpha value = {}".format(alpha))
    print("train score: {:.2f}".format(lasso_x.score(X_train, y_train)))
    print("test score: {:.2f}".format(lasso_x.score(X_test, y_test)))
    print("number of features used: {:.2f}".format(np.sum(lasso_x.coef_ != 0)))
    print("..." * 10)
    lasso_models.append(lasso_x)

for i, model in enumerate(lasso_models):
    label = "Lasso alpha = {}".format(alphas[i])
    plt.plot(model.coef_, 'o', label=label)

plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
plt.hlines(0, 0, len(lasso_models[0].coef_))
plt.ylim(-1.2, 1.2)
plt.title('Lasso Regression Coefficients for Different Alpha Values')
plt.legend()

plt.show()

'''
     y = w[0] * x[0] + ... + w[p] * x[p] + b
        > p is the number of features
        > w, b are parameters of the model that are learned
        > w also called coefficient. it is stored in the 'coef_' attribute
        > b also called intercept. it is stored in the 'intercept_' attribute
        > y is the prediction the model makes

> we want the magnitude of coefficients to be as small as possible.
> regularization means explicitly restricting a model to avoid overfitting.
'''
