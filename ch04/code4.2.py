from sklearn import preprocessing
import numpy as np

X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
    ])
print('原始X：')
print(X)

print('scale缩放：')
print(preprocessing.scale(X))

print('StandardScaler()变换：')
print(preprocessing.StandardScaler().fit_transform(X))

# 使用RobustScaler可避免异常值对均值、方差造成的负面影响
print('RobustScaler()变换：')
print(preprocessing.RobustScaler().fit_transform(X))
