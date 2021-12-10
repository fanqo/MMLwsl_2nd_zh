# 导入numpy
import numpy as np

# 用大写字母表示矩阵
X = np.array([[6], [8], [10], [14], [18]]).reshape(-1,1)  
x_bar = X.mean()   # 均值
print(x_bar)

# 计算方差，直接用numpy库中var，需要将ddof设为1
print(np.var(X, ddof=1))
