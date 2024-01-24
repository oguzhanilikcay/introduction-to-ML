import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd


citibike = mglearn.datasets.load_citibike()
print("citi bike data:\n{}".format(citibike.head()))

plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')

plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha='left')
plt.plot(citibike, linewidth=1)
plt.xlabel('date')
plt.ylabel('rentals')

plt.show()


#
y = citibike.values
X = citibike.index.strftime("%s").astype("int").values.reshape(-1, 1)
n_train = 184


def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print("test R-square score: {:.2f}".format(score))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10, 4))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90, ha='left')

    plt.plot(range(n_train), y_train, label='train')
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label='test')
    plt.plot(range(n_train), y_pred_train, '--', label='prediction train')
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label='prediction test')

    plt.legend(loc=(1.01, 0))
    plt.xlabel('Date')
    plt.ylabel('Rentals')

    plt.show()


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

print("\n(random forest), POSIX time")
eval_on_features(X, y, regressor)


X_hour = citibike.index.hour.values.reshape(-1, 1)
print("\n(random forest), hour of day")
eval_on_features(X_hour, y, regressor)


X_hour_week = np.hstack([
    citibike.index.dayofweek.values.reshape(-1, 1),
    citibike.index.hour.values.reshape(-1, 1)
])

print("\n(random forest), day of week and hour of day")
eval_on_features(X_hour_week, y, regressor)


from sklearn.linear_model import LinearRegression
print("\n(linear regression), day of week and hour of day")
eval_on_features(X_hour_week, y, LinearRegression())


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
X_hour_week_ohe = ohe.fit_transform(X_hour_week).toarray()


from sklearn.linear_model import Ridge
print("\n(ridge), one hot encoded, day of week and hour of day")
eval_on_features(X_hour_week_ohe, y, Ridge())


from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_ohe_poly = poly_transformer.fit_transform(X_hour_week_ohe)
ridge = Ridge()

print("\n(ridge), one hot encoded, degree=2, day of week and hour of day")
eval_on_features(X_hour_week_ohe_poly, y, ridge)


#
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
features = day + hour

features_poly = poly_transformer.get_feature_names_out(features)
features_nonzero = np.array(features_poly)[ridge.coef_ != 0]
coef_nonzero = ridge.coef_[ridge.coef_ != 0]

plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel('feature magnitude')
plt.ylabel('feature')
plt.title('coefficients of the linear regression model using a product of hour and day')

plt.show()
