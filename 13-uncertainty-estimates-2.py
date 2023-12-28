import numpy as np

# uncertainty in multiclass classification
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42
)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbc.fit(X_train, y_train)

print("Decision function shape: {}".format(gbc.decision_function(X_test).shape))
print("Decision function:\n{}".format(gbc.decision_function(X_test)[:6, :]))

print("argmax of decision function:\n{}".format(np.argmax(gbc.decision_function(X_test), axis=1)))
print("predictions:\n{}".format(gbc.predict(X_test)))

print("predicted probabilities:\n{}".format(gbc.predict_proba(X_test)[:6]))
print("sum of predicted probabilities: {}".format(gbc.predict_proba(X_test)[:6].sum(axis=1)))

print("argmax of predicted probabilities:\n{}".format(np.argmax(gbc.predict_proba(X_test), axis=1)))
print("predictions:\n{}".format(gbc.predict(X_test)))



#
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
named_target = iris.target_names[y_train]
logr.fit(X_train, named_target)

print("unique classes in training data: {}".format(logr.classes_))
print("predictions: {}".format(logr.predict(X_test)[:10]))

argmax_dec_func = np.argmax(logr.decision_function(X_test), axis=1)

print("argmax of decision function: {}".format(argmax_dec_func[:10]))
print("argmax combined with classes_: {}".format(logr.classes_[argmax_dec_func][:10]))
