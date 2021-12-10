import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1 表示男性; 0 表示女性；与书中表格中数据有出入
X_train = np.array([
    [158, 1],
    [170, 1],
    [183, 1],
    [191, 1],
    [155, 0],
    [163, 0],
    [180, 0],
    [158, 0],
    [170, 0]
    ])

y_train = [64, 86, 84, 80, 49, 59, 67, 54, 67]

X_test = np.array([
    [168, 1],
    [180, 1],
    [160, 0],
    [169, 0]
    ])

y_test = [65, 96, 52, 67]

# 对数据进行缩放
ss = StandardScaler()   #  z = (x - u) / s，标准化
X_train_scaled = ss.fit_transform(X_train)
print('原始训练数据：')
print(X_train)
print('缩放后数据：')
print(X_train_scaled)

X_test_scaled = ss.fit_transform(X_test)
K = 3
clf = KNeighborsRegressor(n_neighbors=K)
clf.fit(X_train_scaled, y_train)   # y不需要缩放
predictions = clf.predict(X_test_scaled)  # 预测数据也需要缩放
print('预测的权重： %s' % predictions)
print('判定系数： %s' % r2_score(y_test, predictions))

# MAE = 1/n \sum_{i=0}^{n-1} \abs{y_i - \hat{y}_i}
print('平均绝对误差： %s' % mean_absolute_error(y_test, predictions))

# MSE = 1/n \sum_{i=0}^{n-1} (y_i - \hat{y}_i)^2
print('均方偏差： %s' %  mean_squared_error(y_test, predictions))
