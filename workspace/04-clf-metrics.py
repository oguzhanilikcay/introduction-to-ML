import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings

#
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.set_option("display.precision", 2)
pd.set_option("display.colheader_justify", 'center')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
pd.options.display.float_format = '{:.2f}'.format


#
digits = load_digits()
X = digits.data
y = digits.target

print("X.shape: {}\ny.shape:{}".format(X.shape, y.shape))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)


logr = LogisticRegression(C=10)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy_score: {}".format(accuracy))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n{}".format(cm))

c_rep = classification_report(y_test, y_pred)
print("classification report:\n{}".format(c_rep))
