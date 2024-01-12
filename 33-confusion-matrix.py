import matplotlib.pyplot as plt
import numpy as np
import mglearn



from sklearn.datasets import load_digits
digits = load_digits()
y = digits.target == 9

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0
)

from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent')
dummy_majority.fit(X_train, y_train)

pred_most_frequent = dummy_majority.predict(X_test)

print("unique predicted label: {}".format(np.unique(pred_most_frequent)))
print("test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(X_train, y_train)

pred_tree = dtc.predict(X_test)
print("test score tree: {:.2f}".format(dtc.score(X_test, y_test)))


dummy = DummyClassifier()
dummy.fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=0.1, max_iter=500)
logr.fit(X_train, y_train)
pred_logr = logr.predict(X_test)
print("logr score: {:.2f}".format(logr.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_logr)
print("confusion matrix:\n{}".format(cm))


mglearn.plots.plot_confusion_matrix_illustration()
plt.show()

mglearn.plots.plot_binary_confusion_matrix()
plt.show()


# compare the fitted models

print("most frequent class:\n", confusion_matrix(y_test, pred_most_frequent))
print("dummy model:\n", confusion_matrix(y_test, pred_dummy))
print("decision tree:\n", confusion_matrix(y_test, pred_tree))
print("logistic regression:\n", confusion_matrix(y_test, pred_logr))

from sklearn.metrics import f1_score
print("f1 score most frequent: {:.2f}".format(f1_score(y_test, pred_most_frequent)))
print("f1 score dummy model: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("f1 score decision tree: {:.2f}".format(f1_score(y_test, pred_tree)))
print("f1 score logr: {:.2f}".format(f1_score(y_test, pred_logr)))

from sklearn.metrics import classification_report
c_report = classification_report(
    y_test, pred_most_frequent, target_names=['not nine', 'nine'], zero_division=1
)
print("> classification report:\n{}".format(c_report))

c_report_dummy = classification_report(
    y_test, pred_dummy, target_names=['not nine', 'nine'], zero_division=1
)
print("> classfication report (dummy):\n{}".format(c_report_dummy))

c_report_logr = classification_report(
    y_test, pred_logr, target_names=['not nine', 'nine'], zero_division=1
)
print("> classification report (logr):\n{}".format(c_report_logr))
