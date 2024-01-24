import matplotlib.pyplot as plt

#
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(
    X, random_state=5, test_size=0.1
)


# plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c='blue', label='Training set', s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1],
                marker='^', c='red', label='Test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('Original Data')

# scale the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c='blue', label='Training set', s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
                marker='^', c='red', label='Test set', s=60)
axes[1].set_title('Scaled Data')


# visulize wrongly scaled data
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c='blue', label='training set', s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                marker='^', c='red', label='test set', s=60)
axes[2].set_title('Improperly Scaled Data')


for ax in axes:
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

plt.show()

'''
> iki farkli scaler kullanildigi icin egitim ve test setlerinin olcekleri birbirinden farkli olacaktir.
> bu durumda model egitildikten sonra test setindeki veriler egitim sirasinda kullanilan olcekleme
parametrelerine gore degil, kendi olcekleme parametreleriyle islem gorecektir.
'''

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=3
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=4, max_iter=5000)
model.fit(X_train, y_train)

print("train set accuracy: {:.2f}".format(model.score(X_train, y_train)))
print("test set accuracy: {:.2f}".format(model.score(X_test, y_test)))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

model.fit(X_train_scaled, y_train)
print("scaled train set accuracy: {:.2f}".format(model.score(X_train_scaled, y_train)))
print("scaled test set accuracy: {:.2f}".format(model.score(X_test_scaled, y_test)))
