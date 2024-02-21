import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.set_option("display.precision", 2)
pd.set_option("display.colheader_justify", 'center')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
pd.options.display.float_format = '{:.2f}'.format


# dataset (type 2)
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=2, cluster_std=[5.0, 1.25], random_state=30)

print("shape of X: {}".format(X.shape))
print("shape of y: {}".format(y.shape))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.25
)

from sklearn.svm import SVC
svm = SVC(kernel='rbf', probability=True)

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("train score {:.2f}".format(svm.score(X_train, y_train)))
print("test score: {:.2f}".format(svm.score(X_test, y_test)))

'''
> restricted to the binary classification case:
precision_recall_curve
roc_curve
class_likelihood_ratios

> also work in the multiclass case:
roc_auc_score
confusion_matrix


> also work in the multilabel case:
classification_report
accuracy_score
f1_score
multilabel_confusion_matrix
roc_auc_score


for classification:
    > accuracy
    > roc_auc
    > average_precision
    > f1, f1_macro, f1_micro, f1_weighted
for regression:
    > r2
    > mean_squared_error
    > mean_absolute_error
'''


from sklearn.metrics import precision_recall_curve
y_score = svm.decision_function(X_test)
precision, recall, thold = precision_recall_curve(y_test, y_score)

print("\n<y_score>\nshape: {}\nmin value: {:.2f}\nmax value: {:.2f}\n".format(
    y_score.shape, y_score.min(), y_score.max()
))


close_zero = np.argmin(np.abs(thold))
plt.plot(precision[close_zero], recall[close_zero],
         'o', markersize=20, label='thold zero', fillstyle='none', c='k', mew=2)
label = 'precision recall curve'
plt.plot(precision, recall, label=label)
plt.xlabel('precision')
plt.ylabel('recall')
plt.title(label)

plt.show()


from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_test, y_score)
print("average precision : {:.3f}".format(pr_auc))

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_score)
print("roc_auc_score: {:.3f}\n".format(roc_auc))

'''
ROC curve : 
    > The ROC curve is a graphical representation of the trade-off between 'sensitivity (tpr)
    and specificity (fpr)' for a binary classifier at different classification thresholds.
    > The area under the ROC curve (AUC-ROC) is a commonly used metric for evaluating the
    performance of binary classifiers.
    > roc_auc_score()
PR curve :
    > The PR curve is graphical representation of the trade-off betweeen 'precision and recall'
    for a binary classifier at different classification thresholds.
    > The area under the PR curve (AUC-PR) is another commonly used metric for evaluating the
    performance of binary classifiers.
    > average_precision_score()
    
< Difference Between AUC-ROC and AUC-PR >
    > Sensitivity vs. Precision :
        > sensitivity measures how well the model can detect positive cases.
        > precision measures how well the model can detect false positives.
    
    > Imbalanced Data :
        > AUC-ROC is less sensitive to class imbalance than AUC-PR .

    > Interpretation :
        > AUC-ROC measures the model's ability to distinguish between positive and 
        negative cases.
        > AUC-PR measures the model's ability to predict positive cases correctly 
        at all levels of recall.
        
    > Use Cases : 
        > AUC-ROC is a good metric to use when the cost of FP and FN is roughly equal, or
        when the distribution of positive and negative instances is roughly balanced.
        > AUC-PR is more suitable when the cost of FP and FN is highly asymmetric, or
        when the positive class is rare.
'''

'''
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html#sphx-glr-auto-examples-preprocessing-plot-discretization-classification-py
https://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html#sphx-glr-auto-examples-exercises-plot-iris-exercise-py
'''


# analyze decision_function and predict_proba
y_score = svm.decision_function(X_test)
y_proba = svm.predict_proba(X_test)

print("y_score shape: {}".format(y_score.shape))
print("y_proba shape: {}".format(y_proba.shape))
print("y_pred shape: {}\n".format(y_pred.shape))


score_data = np.hstack([y_score.reshape(-1, 1), y_pred.reshape(-1, 1)])

df_score = pd.DataFrame(
    data=score_data,
    columns=['decision_function', 'y_pred']
)
print("decision_function vs y_pred:\n", df_score.head(8))
''' Negative values belong to class 0, while positives belong to class 1.
    A higher absolute value means a more distinct decision and higher confidence in the model.'''


proba_data = np.hstack([y_proba, y_pred.reshape(-1, 1)])

df_proba = pd.DataFrame(
    data=proba_data,
    columns=['probability (class 0)', 'probability (class 1)', 'y_pred']
)
print("predict_proba vs y_pred:\n", df_proba.head(8))
''' compute probabilities of possible outcomes for samples in X '''

arr = np.hstack([y_proba, y_score.reshape(-1, 1)])
df_compare = pd.DataFrame(
    data=arr,
    columns=['proba0', 'proba1', 'decision']
)
print("relation between predict_proba and decision_function:\n", df_compare.head(8))

score_neg = df_compare[df_compare['decision'] < 0]
score_pos = df_compare[df_compare['decision'] > 0]

print("(class 0):\n", score_neg.tail())
print("(class 1):\n", score_pos.tail())

print("\ny_score values (min, max): ({:.2f}, {:.2f})\n".format(
    y_score.min(), y_score.max()
))

n_class0_proba = len([i for i in y_proba[:, 0] if i > 0.5])
n_class0_test = len([i for i in y_test if i == 0])
n_class0_pred = len([i for i in y_pred if i == 0])

print("number of class0 in:\ny_proba: {}\ny_test: {}\ny_pred: {}".format(
    n_class0_proba, n_class0_test, n_class0_pred
))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n{}".format(cm))
