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

# 准确率，所有预测中，预测对的数目占总预测数目的比例
from sklearn.metrics import accuracy_score
print('准确率： %s' % accuracy_score(y_test_binarized, predictions_binarized))

# 精确率，查准率，从预测角度出发，被预测为正向的结果中，真正正向占的比例
from sklearn.metrics import precision_score
print('精确率： %s' % precision_score(y_test_binarized, predictions_binarized))

# 召回率，查全率，从真实情况出发，所有真正正向的情况中，被预测到了的比例
from sklearn.metrics import recall_score
print('召回率： %s' % recall_score(y_test_binarized, predictions_binarized))

# F1得分，精确率、召回率乘积的2倍除以它们的和
from sklearn.metrics import f1_score
print('F1得分： %s' % f1_score(y_test_binarized, predictions_binarized))
