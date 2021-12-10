import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = \
    [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pprint

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)
# 对“列”预测
yy = regressor.predict(xx.reshape(xx.shape[0], 1))

plt.plot(xx, yy)

# 生成特征矩阵，第一列为1
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.fit_transform(X_test)

regressor_quadratic = LinearRegression()   # 同样的方法实例化
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')

plt.title('披萨价格在直径上回归')
plt.xlabel('直径（英寸）')
plt.ylabel('价格（美元）')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show(block=False)

print('原始训练数据X_train：')
print('\t', X_train)

print('拟合、转换后训练数据X_train_quadratic：')
print( '\t' + str(X_train_quadratic).replace('\n', '\n\t'))

print('原始测试数据X_test：')
print('\t', X_test)

print('转换后的训练数据X_test_quadratic：')
print('\t' + str(X_test_quadratic).replace('\n', '\n\t'))

print('简单线性回归R^2： ', regressor.score(X_test, y_test))
print('二次线性回归R^2： ', regressor_quadratic.score(X_test_quadratic, y_test))



