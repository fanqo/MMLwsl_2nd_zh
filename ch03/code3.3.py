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

x = np.array([[155, 70]])
prediction_binarized = clf.predict(x.reshape(1,-1))[0]

# 逆向转换，将整数标签再转换为字符串
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label)
