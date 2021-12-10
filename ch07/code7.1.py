from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = \
    [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']

# Load and return the breast cancer wisconsin dataset (classification)
X, y = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 31)

lr = LogisticRegression(solver='lbfgs')
nb = GaussianNB()

lr_scores = []
nb_scores = []

train_sizes = range(10, len(X_train), 25)

for train_size in train_sizes:
    X_slice, _, y_slice, _ = train_test_split(
        X_train, y_train, train_size = train_size, stratify = y_train,
        random_state = 31)

    nb.fit(X_slice, y_slice)
    nb_scores.append(nb.score(X_test, y_test))
    lr.fit(X_slice, y_slice)
    lr_scores.append(lr.score(X_test, y_test))


plt.plot(train_sizes, nb_scores, label='朴素Bayes')
plt.plot(train_sizes, lr_scores, linestyle='--', label='逻辑回归')
plt.title('朴素Bayes和逻辑回归准确率')
plt.xlabel('训练实例个数')
plt.ylabel('测试集精度')
plt.legend()
plt.show()

