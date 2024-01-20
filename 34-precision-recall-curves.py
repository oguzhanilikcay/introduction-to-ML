import matplotlib.pyplot as plt
import numpy as np
import mglearn

from mglearn.datasets import make_blobs
X, y = make_blobs(
    n_samples=(400, 50), cluster_std=[7.0, 2], random_state=22
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

from sklearn.svm import SVC
svm = SVC(gamma=0.05)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

mglearn.plots.plot_decision_threshold()
plt.show()

from sklearn.metrics import classification_report
c_report = classification_report(y_test, y_pred)
print("classification_report:\n{}".format(c_report))

y_pred_lower_threshold = svm.decision_function(X_test) > -0.8
c_report_threshold = classification_report(y_test, y_pred_lower_threshold)
print("classification_report (with threshold):\n{}".format(c_report_threshold))

print("---"*24)

'''
let's assume in our application it is more important to have a high recall for class 1.
this means we are willing to risk more false positives (false class 1) in exchance for more
true positives (which will increase the recall).
'''

#
# precision, recall, threshold | ROC - AUC
#

from sklearn.metrics import precision_recall_curve
precision0, recall0, thresholds0 = precision_recall_curve(
    y_test, svm.decision_function(X_test)
)


#
X, y = make_blobs(
    n_samples=(4000, 500), cluster_std=[7.0, 2], random_state=22
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

svc = SVC(gamma=0.05)
svc.fit(X_train, y_train)

precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test)
)

close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero],
         'o', markersize=10, label='threshold zero', fillstyle='none', c='k', mew=2)

plt.plot(precision, recall, label='precision recall curve')
plt.xlabel('precision')
plt.ylabel('recall')
plt.title('precison recall curve for SVC(gamma=0.05)')

plt.show()


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rfc.fit(X_train, y_train)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rfc.predict_proba(X_test)[:, 1]
)

plt.plot(precision, recall, label='svc')
plt.plot(precision[close_zero], recall[close_zero],
         'o', markersize=10, label='threshold zero svc', fillstyle='none', c='k', mew=2)
plt.plot(precision_rf, recall_rf, label='random forest')

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf],
         '^', markersize=10, label='threshold 0.5 rfc', fillstyle='none', c='k', mew=2)

plt.xlabel('precision')
plt.ylabel('recall')
plt.legend(loc='best')
plt.title('comparing precision recall curves of svm and random forest')

plt.show()


from sklearn.metrics import f1_score
print("f1 - svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))
print("f1 - rfc: {:.3f}".format(f1_score(y_test, rfc.predict(X_test))))


from sklearn.metrics import average_precision_score
aps_svc = average_precision_score(y_test, svc.decision_function(X_test))
aps_rfc = average_precision_score(y_test, rfc.predict_proba(X_test)[:, 1])

print("average precision - svc: {:.3f}".format(aps_svc))
print("average precision - rfc: {:.3f}".format(aps_rfc))
''' average precision is the area under a curve that goes from 0 to 1. '''


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR (recall)')

close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero],
         'o', markersize=10, label='threshold zero', fillstyle='none', c='k', mew=2)
plt.legend(loc=4)
plt.title('ROC curve for svm')

plt.show()


fpr_rf, tpr_rf, thold_rf = roc_curve(y_test, rfc.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label='ROC curve svc')
plt.plot(fpr_rf, tpr_rf, label='ROC curve rf')

plt.plot(fpr[close_zero], tpr[close_zero],
         'o', markersize=10, label='threshold close zero', fillstyle='none', c='k', mew=2)

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf], tpr_rf[close_default_rf],
         '^', markersize=10, label='threshold 0.5 rf', fillstyle='none', c='k', mew=2)

plt.legend(loc=4)
plt.xlabel('FPR')
plt.ylabel('TPR (recall)')
plt.title('comparing ROC curves for svm and random forest')

plt.show()


from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))

print("AUC for rfc: {:.2f}".format(rf_auc))
print("AUC for svc: {:.2f}".format(svc_auc))
