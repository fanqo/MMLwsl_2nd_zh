import numpy as np

# ||u - v||_2, \sqrt{\sum (u_i - v_i)^2 }
from scipy.spatial.distance import euclidean

# 身高，单位 mm
X_train = np.array([
    [1700, 1],
    [1600, 0]
    ])

x_test = np.array([1640, 1]).reshape(1,-1)
print('身高单位为mm时距离：')
print(euclidean(X_train[0, :], x_test))
print(euclidean(X_train[1, :], x_test))

# 身高，单位 m
X_train = np.array([
    [1.7, 1],
    [1.6, 0]
    ])

x_test = np.array([1.64, 1]).reshape(1, -1)
print('身高单位为m时距离：')
print(euclidean(X_train[0, :], x_test))
print(euclidean(X_train[1, :], x_test))


## 对第二个x_test的输入及书上的输出有疑问
