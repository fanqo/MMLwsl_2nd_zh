import numpy as np

# Note： 此处数据与表中有出入
X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
    ])

y_train = ['male', 'male', 'male', 'male', 'female', 'female',
           'female', 'female', 'female']

x = np.array([[155, 70]])
# np.sum() 中 axis=1，表示sum[i] = a[i][0]+a[i][1]
distances = np.sqrt(np.sum((X_train - x)**2, axis=1))

nearest_neighbor_indices = distances.argsort()[:3]
nearest_neighbor_genders = np.take(y_train, nearest_neighbor_indices)

from collections import Counter
b = Counter(nearest_neighbor_genders)

# most_common(n) 列出前n个最常见的元素
print('预期性别: %s' % b.most_common(1)[0][0])

