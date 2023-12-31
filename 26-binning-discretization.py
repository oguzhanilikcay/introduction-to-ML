import matplotlib.pyplot as plt
import numpy as np
import mglearn

#
X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_split=3)
dtr.fit(X, y)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)


plt.plot(line, dtr.predict(line), label='decision tree regression')
plt.plot(line, lr.predict(line), label='linear regression')
plt.plot(X[:, 0], y, 'o', c='k')

plt.ylabel('Regression output')
plt.xlabel('Input Feature')
plt.legend(loc='best')
plt.title('comparing linear regression and a decision tree on the wave dataset')

plt.show()


#
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)

print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
ohe.fit(which_bin)
X_binned = ohe.transform(which_bin)

print("X_binned:\n{}".format(X_binned[:5]))
print("X_binned.shape: {}".format(X_binned.shape))


# build lr and dtr model on the ohe'ed data.
# visualize with the bin boundaries
line_binned = ohe.transform(np.digitize(line, bins=bins))
'''line dizisindeki her bir degeri bins bolgesindeki sinirlara gore bir bolgeye atar.'''
lr = LinearRegression()
lr.fit(X_binned, y)
plt.plot(line, lr.predict(line_binned), label='linear regression binned')

dtr = DecisionTreeRegressor(min_samples_split=3)
dtr.fit(X_binned, y)
plt.plot(line, dtr.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')

plt.vlines(bins, -3, 3, linewidth=1, alpha=0.2)
plt.legend(loc='best')
plt.ylabel('Regression output')
plt.xlabel('Input feature')

plt.show()


#
X_combined = np.hstack([X, X_binned])
print("X_combined.shape: {}".format(X_combined.shape))

lr = LinearRegression()
lr.fit(X_combined, y)

lines_combined = np.hstack([line, line_binned])
plt.plot(line, lr.predict(lines_combined), label='linear regression combined')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.legend(loc='best')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.title('lr using binned features and a sine global slope')
plt.plot(X[:, 0], y, 'o', c='k')

plt.show()


#
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)

lr = LinearRegression()
lr.fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, lr.predict(line_product), label='linear regression product')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.title('lr with a separate slope per bin')
plt.legend(loc='best')

plt.show()

''' her bir bolge icin ayri bir egimle modelleme yapabilme esnekligi saglandi. veririn belli
bolgelerde farkli davranislar sergiledigi durumlar ele alindi'''



#
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)

X_poly = poly.transform(X)
print("X_poly.shape: {}".format(X_poly.shape))
print("X.shape: {}".format(X.shape))

print("entries of X:\n{}".format(X[:5]))
print("entries of X_poly:\n{}".format(X_poly[:5]))

print("polynomial feature names:\n{}".format(poly.get_feature_names_out()))


reg = LinearRegression()
reg.fit(X_poly, y)
line_poly = poly.transform(line)

plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.ylabel('Inpurt feature')
plt.title('lr with tenth-degree polynomial features')
plt.legend(loc='best')

plt.show()


#
from sklearn.svm import SVR

for gamma in [1, 10]:
    svr = SVR(gamma=gamma)
    svr.fit(X, y)
    plt.plot(line, svr.predict(line), label="SVR gamma={}".format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.title('comparison of different gamma paramters for an SVM with RBF kernel')
plt.legend(loc='best')

plt.show()
