from numpy.linalg import inv
from numpy import dot, transpose

# 解释变量，为了考虑截距，第一列为1
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
# 响应变量
y = [[7], [9], [13], [17.5], [18]]

beta = dot(inv(dot(transpose(X), X)), dot(transpose(X), y))
print(beta)


