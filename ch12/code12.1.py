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

    
    
