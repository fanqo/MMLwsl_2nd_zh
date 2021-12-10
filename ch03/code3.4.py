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

from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

# 将字符串标签转换为整数
lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)

K = 3   # 使用K个最邻近点
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))

X_test = np.array([
    [168,65],
    [180,96],
    [160,52],
    [169,67]
    ])

y_test = ['male', 'male', 'female', 'female']
y_test_binarized = lb.transform(y_test)
print('二元化的标签： %s' % y_test_binarized.T[0])
predictions_binarized = clf.predict(X_test)
print('二元化的预测： %s' % predictions_binarized)
print('预测的标签： %s' % lb.inverse_transform(predictions_binarized))
