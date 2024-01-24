import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Decision Tree
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=4, random_state=0)
dtc.fit(X_train, y_train)
''' limiting the depth of the tree decreases overfitting. this leads to a lower accuray on the training set, but an
improvement on the test set'''

print("train set accuracy: {:.3f}".format(dtc.score(X_train, y_train)))
print("test set accuracy: {:.3f}".format(dtc.score(X_test, y_test)))


# analyzing decision trees
from sklearn.tree import export_graphviz
export_graphviz(dtc, out_file='../dtc.dot', class_names=['malignant', 'benign'],
                feature_names=cancer.feature_names, impurity=False, filled=True)

# visulize the tree
with open('../dtc.dot') as f:
    dot_graph = f.read()

import graphviz
graphviz.Source(dot_graph)
''' console command:>> dot -Tpng dtc.dot -o dtc.png '''

print("Feature importances:\n{}".format(dtc.feature_importances_))


# visualize the coefficients
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title('Feature importance table')

    plt.show()

plot_feature_importances_cancer(dtc)


# dataset - ram_prices
ram_prices = pd.read_csv('../csv-files/ram_price.csv')
print(ram_prices.head())

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('Year')
plt.ylabel('Price in $/Mbyte')

plt.show()


data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train['date'].to_numpy()
y_train = np.log(data_train.price).to_numpy()

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


X_all = np.array(ram_prices.date).reshape(-1, 1)

pred_dtr = dtr.predict(X_all)
pred_lr = lr.predict(X_all)

price_dtr = np.exp(pred_dtr)
price_lr = np.exp(pred_lr)


plt.semilogy(data_train.date, data_train.price, label='Training data')
plt.semilogy(data_test.date, data_test.price, label='Test data')
plt.semilogy(ram_prices.date, price_dtr, label='Decision tree prediction')
plt.semilogy(ram_prices.date, price_lr, label='Linear regression prediction')
plt.legend()

plt.show()

''' Decison Tree Regressor (and all other tree-based regression models) is not able to extrapolate, or make
predictions outside of the training data. That means the tree has no ability to generate "new" responses
outside of what was seen in the training data.

The possible splits of the data dont depend on scaling, no preprocessing like normalization or standardization
of features is needed for decision tree algorithms.'''
