# 参考 https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


def g(x):
    if x > 0:
        return 1
    else:
        return 0

def update_weights(w, X, y, a = 1):
    py = 0
    for i in range(X.shape[0]):
        py = g(np.dot(X[i], w))
        w = w + a*(y[i] - py)*X[i]
        print('weights: {}'.format(w))
    return w
    
    
if __name__ == '__main__':
    
    X = np.array([[1.0, 0.2, 0.1],
                  [1.0, 0.4, 0.6],
                  [1.0, 0.5, 0.2],
                  [1.0, 0.7, 0.9]])

    y = np.array([1, 1, 1, 0])
    w = np.array([0, 0, 0])

    n_epoch = 6
    for i in range(n_epoch):
        w = update_weights(w, X, y)
        print('_'*n_epoch, (i+1))

    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = -0.9, 1.9
    y_min, y_max = -1.05, 2.05
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    fig, ax = plt.subplots()
    xyc = np.c_[xx.ravel(), yy.ravel()]
    Z = np.zeros(xyc.shape[0])
    for i in range(len(Z)):
        Z[i] = g(np.dot(xyc[i], w[1:]) + w[0])

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # ax.axis('off')
    ax.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired, edgecolors='blue')
    ax.set_title('Perceptron')
