import pandas as pd


data = pd.read_csv('csv-files/boston_house_prices.csv')

X = data.iloc[:, 0:13].to_numpy()
y = data.iloc[:, 13:14].to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly.fit(X_train_scaled)

X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_poly.shape: {}".format(X_train_poly.shape))
print("polynomial feature names:\n{}".format(poly.get_feature_names_out()))


# impact of interactions with Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(random_state=0)

ridge.fit(X_train_scaled, y_train)
score_scaled = ridge.score(X_test_scaled, y_test)
print("score without interactions(ridge): {:.3f}".format(score_scaled))


ridge.fit(X_train_poly, y_train)
score_poly = ridge.score(X_test_poly, y_test)
print("score with interactions(ridge): {:.3f}".format(score_poly))


# impact of interactions with Random Forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=0)

rfr.fit(X_train_scaled, y_train.ravel())
score_scaled = rfr.score(X_test_scaled, y_test)
print("score without interactions(random forest): {:.3f}".format(score_scaled))

rfr.fit(X_train_poly, y_train.ravel())
score_poly = rfr.score(X_test_poly, y_test)
print("score with interactions(random forest): {:.3f}".format(score_poly))
