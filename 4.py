import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split


# Linear models for multiclass classification

from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2'])
plt.show()

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X, y)

print('Coefficient Shape:\n', model.coef_)
print('Intercept Shape: ', model.intercept_)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

for coef, intercept, color, in zip(model.coef_, model.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2',
            'Line class 0', 'Line class 1', 'Line class 2'],
           loc=(0.92, 0.3))

plt.show()

mglearn.plots.plot_2d_classification(model, X, fill=True, alpha=0.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

for coef, intercept, color in zip(model.coef_, model.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.legend(['Class 0', 'Class 1', 'Class 2',
            'Line class 0', 'Line class 1', 'Line class 2'],
           loc=(0.92, 0.3))
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')


plt.show()












