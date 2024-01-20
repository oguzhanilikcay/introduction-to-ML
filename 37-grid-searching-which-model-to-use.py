import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

#
# Grid-Searching Preprocessing Steps and Model Parameters
#

boston = pd.read_csv('csv-files/boston_house_prices.csv')

X = boston.iloc[:, 0:13].to_numpy()
y = boston.iloc[:, 13:14].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge()
)

param_grid = {
    'polynomialfeatures__degree': [1, 2, 3],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

plt.matshow(gs.cv_results_['mean_test_score'].reshape(3, -1), vmin=0, cmap='viridis')

plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree'])
plt.title("heat map of mean cv score as a function of the degree of the pf and alpha parameter of Ridge")
plt.colorbar()

plt.show()

print("best parameters: {}".format(gs.best_params_))
print("test set score: {:.2f}".format(gs.score(X_test, y_test)))

#
param_grid = {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
gs = GridSearchCV(pipe, param_grid=param_grid, cv=5)
gs.fit(X_train, y_train)

print("test score without polynomial features: {:.2f}".format(gs.score(X_test, y_test)))

#
# Grid-Searching Which Model To Use
#

pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', SVC())
])

param_grid = [
    {
        'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
        'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'classifier': [RandomForestClassifier(n_estimators=100)], 'preprocessing': [None],
        'classifier__max_features': [1, 2, 3]
    }
]

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

gs = GridSearchCV(pipe, param_grid, cv=5)
gs.fit(X_train, y_train)

print("best params:\n{}".format(gs.best_params_))
print("best cross-validation score: {:.2f}".format(gs.best_score_))
print("test score: {:.2f}".format(gs.score(X_test, y_test)))
