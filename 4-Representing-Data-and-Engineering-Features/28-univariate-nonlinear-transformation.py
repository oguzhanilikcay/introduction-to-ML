import matplotlib.pyplot as plt
import numpy as np


rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

bins = np.bincount(X[:, 0])
print("number of feature appearances:\n{}".format(bins))

plt.bar(range(len(bins)), bins, color='grey')
plt.ylabel('number of appearances')
plt.xlabel('value')
plt.title('histogram of feature values for X[0]')

plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
score = ridge.score(X_test, y_test)

print("test score: {:.2f}".format(score))

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='gray')
plt.ylabel('number of appearances')
plt.xlabel('value')
plt.title('histogram of feature values for X[0] after logarithmic transformation')

plt.show()

ridge.fit(X_train_log, y_train)
score_log = ridge.score(X_test_log, y_test)
print("test score(log scaled): {:.2f}".format(score_log))
