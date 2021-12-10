import numpy as np

X = np.array([[1, -2],
              [2, -3]
])
w, v = np.linalg.eig(X)
# w 特征值， v 特征向量

np.set_printoptions(formatter={'float': '{:0.3f}'.format})
print(w)
print(v)
