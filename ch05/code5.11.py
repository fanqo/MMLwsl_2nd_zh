import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# load_boston 载入并返回 boston 房价数据集
# data.data, data.target, data.filename
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_train = y_train.ravel()
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test.reshape(-1, 1))

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print('交叉验证R^2: {}'.format(scores))
print('交叉验证平均R^2: {}'.format(scores.mean()))

# 虽然前面进行了交叉验证，还要重新fit
regressor.fit(X_train,  y_train)
print('测试集R^2: {}'.format(regressor.score(X_test, y_test)))
