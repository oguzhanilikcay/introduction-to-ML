import pandas as pd


data = pd.read_csv('../csv-files/housing_price_dataset.csv')
print("> data first 5 rows:\n{}".format(data.head()))
print("> column names:\n{}".format(list(data.columns)))

data_dummies = pd.get_dummies(data.loc[:10000])
print("> data_dummies view:\n {}".format(data_dummies.iloc[:, 5:8].head()))
print("> data_dummies column names:\n {}".format(list(data_dummies.columns)))


X = data_dummies.iloc[:, [0, 1, 2, 3, 5, 6, 7]].values
y = data_dummies['Price'].values

print("X.shape: {} , y.shape: {}".format(X.shape, y.shape))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=10, test_size=0.20, shuffle=True
)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("train score: {:.2f}".format(lr.score(X_train, y_train)))
print("test score: {:.2f}".format(lr.score(X_test, y_test)))
