import numpy as np

X = np.array([[2.0, 0.0, -1.4],
              [2.2, 0.2, -1.5],
              [2.4, 0.1, -1.0],
              [1.9, 0.0, -1.2]
])

np.set_printoptions(formatter={'float': '{:0.3f}'.format})
print(np.cov(X))
