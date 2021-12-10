import numpy as np

X = np.array([
    [0.9, 1.0],
    [2.4, 2.6],
    [1.2, 1.7],
    [0.5, 0.7],
    [0.3, 0.7],
    [1.8, 1.4],
    [0.5, 0.6],
    [0.3, 0.6],
    [2.5, 2.6],
    [1.3, 1.1]
])

C = np.cov(X[:,0], X[:,1])
w, v = np.linalg.eig(C)
# w 特征值， v 特征向量

np.set_printoptions(formatter={'float': '{:0.3f}'.format})
print('协方差矩阵为：')
print(C)

print('协方差矩阵特征值为：')
print(w)
print('\t特征向量为：')
print(v)

print('投影到主成分的结果：')
print(np.dot(X - np.mean(X, axis=0), v[:,0].T))
