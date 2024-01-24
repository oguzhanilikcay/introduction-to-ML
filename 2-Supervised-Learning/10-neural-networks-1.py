import numpy as np
import matplotlib.pyplot as plt
import mglearn

#
line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), label='relu')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('relu(x), tanh(x)')

plt.show()


#
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10], max_iter=10000)
mlp.fit(X_train, y_train)

mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

plt.show()


# two hidden layers, with 10 units each
mlp2 = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10], max_iter=10000)
mlp2.fit(X_train, y_train)

mglearn.plots.plot_2d_separator(mlp2, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

plt.show()


# two hidden layers, with 10 units each, with tanh nonlinearity
mlp3 = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10], max_iter=10000)
mlp3.fit(X_train, y_train)

mglearn.plots.plot_2d_separator(mlp3, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

plt.show()


#
fig, axes = plt.subplots(2, 4, figsize=(20, 8))

for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp0 = MLPClassifier(solver='lbfgs', random_state=0, alpha=alpha,
                             hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], max_iter=10000)

        mlp0.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp0, X_train, fill=True, alpha=0.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)

        ax.set_title("n_hidden[{}, {}]\nalpha={:.4f}".format(
            n_hidden_nodes, n_hidden_nodes, alpha
        ))

plt.show()
