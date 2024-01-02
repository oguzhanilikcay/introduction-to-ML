import matplotlib.pyplot as plt
import numpy as np
import mglearn


# a simple grid search with for loop
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

from sklearn.svm import SVC

# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1
)

print("size of training set: {}\nsize of validation set: {}\nsize of test set: {}".format(
    X_train.shape[0], X_valid.shape[0], X_test.shape[0]
))

best_score = 0

for gamma in C_gamma_list:
    for C in C_gamma_list:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)

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
print("test set score: {:.2f}".format(gs.score(X_test, y_test)))
print("best parameters: {}".format(gs.best_params_))
print("best cross-validation score: {:.2f}".format(gs.best_score_))

print("best estimator:\n{}".format(gs.best_estimator_))
print("\n\ngrid search results:\n{}".format(gs.cv_results_))
