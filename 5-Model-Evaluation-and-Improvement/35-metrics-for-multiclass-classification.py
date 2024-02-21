import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


from sklearn.datasets import load_digits
digits = load_digits()
y = digits.target == 9

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0
)


plt.figure()

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve

for gamma in [1, 0.05, 0.01]:
    svc = SVC(gamma=gamma)
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)

    decision_val = svc.decision_function(X_test)
    auc = roc_auc_score(y_test, decision_val)
    fpr, tpr, _ = roc_curve(y_test, decision_val)

    print("<gamma = {:.2f}> <accuracy = {:.2f}> <AUC = {:.2f}>".format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma= {:.2f}".format(gamma))

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc='best')
plt.title("comparing ROC curves of SVMs with different settings of gamma")

plt.show()

#
# metrics for multiclass classification
#

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=0
)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {:.3f}".format(accuracy))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n{}".format(cm))

import seaborn
seaborn.heatmap(
    cm,
    xticklabels=digits.target_names,
    yticklabels=digits.target_names,
    cmap=plt.cm.gray_r,
    annot=True
)

plt.xlabel("predicted label")
plt.ylabel("true label")
plt.title("confusion matrix")

plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''
if you care about "each sample equally much" it is recommended to use the "micro" average f1 score
if you care abour "each class equally much" it it recommended to use the "macro" average f1 score
'''

from sklearn.metrics import f1_score
print("f1 score - micro average: {:.4f}".format(f1_score(y_test, y_pred, average="micro")))
print("f1 score - macro average: {:.4f}".format(f1_score(y_test, y_pred, average='macro')))


from sklearn.model_selection import cross_val_score
# default scoring for classification is accuracy
print("default scoring: {}".format(
    cross_val_score(SVC(), digits.data, digits.target == 9)))
# providing scoring="accuracy" doesn't change the results
explicit_accuracy = cross_val_score(
    SVC(), digits.data, digits.target == 9, scoring="accuracy"
)
print("explicit accuracy scoring: {}".format(explicit_accuracy))

roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc")
print("AUC scoring: {}".format(roc_auc))


#      we can change the metric used to pick the best parameters in GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target == 9, random_state=0
)

print("X_train:\n{}".format(X_train))
print("target == 9: {}".format(digits.target == 9))


param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}

from sklearn.model_selection import GridSearchCV

# using the default scoring of accuracy;
gs = GridSearchCV(SVC(), param_grid=param_grid)
gs.fit(X_train, y_train)

print("<<<  grid-search with accuracy  >>>")
print("best parameters: ", gs.best_params_)
print("best cross-validation score(accuracy): {:.3f}".format(gs.best_score_))
print("test set AUC: {:.3f}".format(roc_auc_score(y_test, gs.decision_function(X_test))))
print("test set accuracy: {:.3f}".format(gs.score(X_test, y_test)))

# using AUC scoring instead;
gs = GridSearchCV(SVC(), param_grid=param_grid, scoring='roc_auc')
gs.fit(X_train, y_train)

print("<<<  grid-search with AUC  >>>")
print("best parameters:", gs.best_params_)
print("best cross-validation score(AUC): {:.3f}".format(gs.best_score_))
print("test set AUC: {:.3f}".format(
    roc_auc_score(y_test, gs.decision_function(X_test))
))
print("test set accuracy: {:.3f}".format(gs.score(X_test, y_test)))
