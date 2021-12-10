from numpy.linalg import lstsq

# 解释变量，为了考虑截距，第一列为1
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
# 响应变量
y = [[7], [9], [13], [17.5], [18]]

beta = lstsq(X, y)[0]   # [0] 最小二乘法的解， [1] 残差
print(beta)

