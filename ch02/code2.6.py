# 导入numpy
import numpy as np

# 用大写字母表示矩阵
X = np.array([[6], [8], [10], [14], [18]]).reshape(-1,1)  
x_bar = X.mean()   # 均值

# 用小写字母表示矢量
y = np.array([7, 9, 13, 17.5, 18])
y_bar = y.mean()   # 均值

# 计算协方差的观察值，分母为n-1，X要转置
covariance = np.multiply((X - x_bar).transpose(), y - y_bar).sum() \
             / (X.shape[0] - 1)
print('协方差为： %.2f' % covariance)

# np.cov() 返回协方差矩阵
print('协方差为： %.2f' % np.cov(X.transpose(), y)[0][1])
