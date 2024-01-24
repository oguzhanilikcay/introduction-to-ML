import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("train set accuracy: {:.2f}".format(mlp.score(X_train, y_train)))
print("test set accuracy: {:.2f}".format(mlp.score(X_test, y_test)))


# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the mean value per feature on the training set
std_on_train = X_train.std(axis=0)
# subtract the mean, and scale by inverse standard deviation
X_train_scaled = (X_train - mean_on_train) / std_on_train
# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train


#
print("--"*5, "compare different parameters", "--"*5)

# - 0 -
mlp = MLPClassifier(random_state=0, max_iter=250)
mlp.fit(X_train_scaled, y_train)

print("train set accuracy(n): {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("test set accuracy(n): {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# - 1 -
mlp = MLPClassifier(random_state=0, max_iter=1000)
mlp.fit(X_train_scaled, y_train)

print("train set accuracy(max_iter=1000): {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("test set accuracy(max_iter=1000): {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# - 2 -
mlp = MLPClassifier(random_state=0, max_iter=250, alpha=1)
mlp.fit(X_train_scaled, y_train)

print("train set accuracy(alpha=1): {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("test set accuracy(alpha=1): {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# - 3 -
mlp = MLPClassifier(random_state=0, max_iter=1000, alpha=1)
mlp.fit(X_train_scaled, y_train)

print("train set accuracy(max_iter=1000, alpha=1): {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("test set accuracy(max_iter=1000, alpha=1): {:.3f}".format(mlp.score(X_test_scaled, y_test)))


#
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()

plt.show()
