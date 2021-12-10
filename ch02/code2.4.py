# 导入numpy
import numpy as np

# 用大写字母表示矩阵
X = np.array([[6], [8], [10], [14], [18]]).reshape(-1,1)  
x_bar = X.mean()   # 均值
print(x_bar)

# 计算方差，分母为样本数目减去1,构成总体方差的无偏估计
variance = ((X - x_bar)**2).sum() / (X.shape[0] - 1)
print(variance)
