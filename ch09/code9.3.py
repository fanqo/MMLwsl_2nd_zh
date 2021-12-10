# 推进法

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = \
    [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']

X, y = make_classification(n_samples = 1000, n_features = 50,
                           n_informative = 30,
                           n_clusters_per_class = 3,
                           random_state = 11)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state = 11)

clf = DecisionTreeClassifier(random_state = 11)
clf.fit(X_train, y_train)
print('决策树准确率： {}'.format(clf.score(X_test, y_test)))

clf = AdaBoostClassifier(n_estimators = 50, random_state = 11)
clf.fit(X_train, y_train)
accuracies = []
accuracies.append(clf.score(X_test, y_test))

plt.title('集成准确率')
plt.ylabel('准确率')
plt.xlabel('集成中基础估计器的数量')
plt.plot(range(1, 51),
         [accuracy for accuracy in clf.staged_score(X_test, y_test)])

plt.show()
