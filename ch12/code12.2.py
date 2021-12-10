# 逼近XOR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

y = [0, 1, 1, 0]
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

clf = MLPClassifier(solver = 'lbfgs', activation = 'logistic',
                    hidden_layer_sizes = (2, ), random_state = 20)

clf.fit(X, y)

predictions = clf.predict(X)
print('准确率： {}'.format(clf.score(X, y)))

for i, p in enumerate(predictions):
    print('真实值： {}， 预测值： {}'.format(y[i], p))

print('连接输入层和隐层的权重为： \n{}'.format(clf.coefs_[0]))
print('隐层的偏权重为： \n{}'.format(clf.intercepts_[0]))

print('连接隐层和输出层的权重为： \n{}'.format(clf.coefs_[1]))
print('输出层的偏权重为： \n{}'.format(clf.intercepts_[1]))

    
    
