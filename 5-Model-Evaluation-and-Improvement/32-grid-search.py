import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
import seaborn

# a simple grid search with for loop:
from sklearn.datasets import load_iris
iris = load_iris()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

print("size of training set: {}\nsize of test set: {}".format(
    X_train.shape[0], X_test.shape[0]))

best_score = 0
C_gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]

from sklearn.svm import SVC

for gamma in C_gamma_list:
    for C in C_gamma_list:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

print("best score: {:.2f}".format(best_score))
print("best parameters: {}".format(best_parameters))

#
#
#

mglearn.plots.plot_threefold_split()
plt.show()

'''
(X_train, y_train) to build the model
(X_test, y_test) to evaluate the performance
(X_val, y_val) to select the parameters

> any choices made based on the test set accuracy "leak" information from the test set into the
model. Therefore, it is important to keep a seperate test set, which is only used for the final evaluation.
> because we used the test data to adjust the parameters, we can no longer us it to assess how
good the model is.
> (train, test, val) =? (75, 15, 10)
'''
# split <data> into <train+validation> set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0, test_size=0.18
)

# split <train+validation> set into <training and validation> sets
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, random_state=1, test_size=0.12
)

print("size of training set: {}\nsize of validation set: {}\nsize of test set: {}".format(
    X_train.shape[0], X_val.shape[0], X_test.shape[0]
))

best_score = 0

for gamma in C_gamma_list:
    for C in C_gamma_list:

        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_val, y_val)

        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

# rebuild a model on the combined training and validation set, and evalute it on the test set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)

print("best score on validation set: {:.2f}".format(best_score))
print("best parameters:", best_parameters)
print("test set score with best parameters: {:.2f}".format(test_score))

'''
> we used the test data to adjust the parameters, we can no longer use it to assess how good
the model is. we need an independent dataset to evaluate, one that was not used to create the model.

> "training set" : to build the model
> "validation set" : to select the parameters of the model
> "test set" : to evaluate the performance of the selected parameters.
'''

#
# grid search with cross-validation
#

from sklearn.model_selection import cross_val_score

for gamma in C_gamma_list:
    for C in C_gamma_list:

        svm = SVC(gamma=gamma, C=C)

        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)

        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)


mglearn.plots.plot_cross_val_selection()
plt.show()

mglearn.plots.plot_grid_search_overview()
plt.show()

#
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

gs.fit(X_train, y_train)

print("> test set score: {:.2f}".format(gs.score(X_test, y_test)))
print("> best parameters: {}".format(gs.best_params_))
print("> best cross-validation score: {:.2f}".format(gs.best_score_))
print("> combination of parameters: {}".format(gs.param_grid))

print("> best estimator: {}".format(gs.best_estimator_))

# convert cv_results to DataFrame
results = pd.DataFrame(gs.cv_results_)
print(results.tail())
print("column names: {}".format([i for i in results.columns]))

scores = np.array(results.mean_test_score).reshape(6, 6)


seaborn.heatmap(
    scores,
    xticklabels=param_grid['gamma'],
    yticklabels=param_grid['C'],
    cmap="viridis",
    annot=True)

plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('heat map of mean cross-validation score as function of C and gamma')

plt.show()


fig, axes = plt.subplots(1, 3, figsize=(13, 5))

param_grid_linear = {'C': np.linspace(1, 2, 6),
                     'gamma': np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.linspace(1, 2, 6),
                      'gamma': np.logspace(-3, 2, 6)}
param_grid_range = {'C': np.logspace(-3, 2, 6),
                    'gamma': np.logspace(-7, -2, 6)}

for param_grid, ax in zip(
        [param_grid_linear, param_grid_one_log, param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

    seaborn.heatmap(
        scores,
        xticklabels=param_grid['gamma'],
        yticklabels=param_grid['C'],
        cmap='viridis',
        annot=True,
        ax=ax
    )

    plt.xlabel('gamma')
    plt.ylabel('C')

plt.show()


#
param_grid = [
    {'kernel': ['rbf'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100],
     'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'kernel': ['linear'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100]}
]
print("> list of grids:\n{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("> best parameters: {}".format(grid_search.best_params_))
print("> best cross-validation score: {:.2}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
print("> cv_results_:\n{}".format(results.T))

#
# nested cross-validation
#

scores = cross_val_score(
    GridSearchCV(SVC(), param_grid, cv=5),
    iris.data, iris.target, cv=5
)

print("cross-validation score: ", scores)
print("mean cross-validation score: ", scores.mean())

print("---"*30)

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []

    for training_samples, test_samples in outer_cv.split(X, y):
        best_params = {}
        best_score = -np.inf


        for parameters in parameter_grid:
            cv_scores = []

            for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):

                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])

                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = parameters

        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])

        outer_scores.append(clf.score(X[test_samples], y[test_samples]))

    return np.array(outer_scores)


from sklearn.model_selection import ParameterGrid, StratifiedKFold

scores = nested_cv(
    X=iris.data, y=iris.target,
    inner_cv=StratifiedKFold(5),
    outer_cv=StratifiedKFold(5),
    Classifier=SVC,
    parameter_grid=ParameterGrid(param_grid)
)

print("cross-validation scores: {}".format(scores))
