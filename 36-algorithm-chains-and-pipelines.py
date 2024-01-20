import matplotlib.pyplot as plt
import numpy as np
import mglearn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("X_train_scaled:\n{}".format(X_train_scaled))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)

score = svm.score(X_test_scaled, y_test)
print("test score: {:.2f}".format(score))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

gs = GridSearchCV(
    SVC(), param_grid=param_grid, cv=5
)
gs.fit(X_train_scaled, y_train)

print("best cross-validation accuracy: {:.2f}".format(gs.best_score_))
print("best test score: {:.2f}".format(gs.score(X_test_scaled, y_test)))
print("best parameters: ", gs.best_params_)

mglearn.plots.plot_improper_processing()
plt.show()


#       < Building Pipelines >

from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
'''pipe.fit first calls fit on the first step (the scaler), then transforms the training data
using the scaler, and finally fits the SVM with the scaled data.'''

print("test score: {:.2f}".format(pipe.score(X_test, y_test)))


# using pipeline in Grid Search:

param_grid = {
    'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=5
)
grid.fit(X_train, y_train)

print("best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("best parameters: {}".format(grid.best_params_))

''' In contrast to the grid search we did before, now for each split in the cross-validation,
the MinMaxScaler is refit with only the training splits and no information is leaked from the 
test tsplit into the parameter seach'''

mglearn.plots.plot_proper_processing()
plt.show()


# Illustratin Information Leakage:
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100, ))

from sklearn.feature_selection import SelectPercentile, f_regression
select = SelectPercentile(score_func=f_regression, percentile=5)
select.fit(X, y)

X_selected = select.transform(X)
print("X_selected.shape: {}".format(X_selected.shape))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

cv_score = cross_val_score(Ridge(), X_selected, y, cv=5)
print("mean cross-validation score (cv only on ridge): {:.2f}".format(np.mean(cv_score)))


pipe = Pipeline([
    ("select", SelectPercentile(score_func=f_regression, percentile=5)),
    ("ridge", Ridge())
])

cv_score = cross_val_score(pipe, X, y, cv=5)
print("mean cross-validation score (pipeline): {:.2f}".format(np.mean(cv_score)))


# The General Pipeline Interface:

def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:, -1]:
        # iterate over all but the final step
        # fit and transform the data
        X_transformed = estimator.fit_transform(X_transformed, y)
    # fit the last step
    self.steps[-1][1].fit(X_transformed, y)

    return self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # iterate over all but the efinal step
        # transform the data
        X_transformed = step[1].transform(X_transformed)
    # fit the last step

    return self.steps[-1][1].predict(X_transformed)


# Concenient Pipeline Creation with make_pipeline
from sklearn.pipeline import make_pipeline

# standard syntax:
pipe_long = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svm", SVC(C=100))
])

# abbreviated syntax:
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

print("pipe_short steps:\n{}".format(pipe_short.steps))


#
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print("pipeline steps:\n{}".format(pipe.steps))


# Accessing Step Attributes
pipe.fit(cancer.data)
components = pipe.named_steps["pca"].components_
print("components.shape: {}".format(components.shape))


#
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=4
)

gs = GridSearchCV(pipe, param_grid, cv=5)
gs.fit(X_train, y_train)

print("best estimator:\n{}".format(gs.best_estimator_))

print("logistic regression step:\n{}".format(
    gs.best_estimator_.named_steps["logisticregression"]
))
print("logistic regression coefficients:\n{}".format(
    gs.best_estimator_.named_steps["logisticregression"].coef_
))
