import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_mldata
# from sklearn.datasets import fetch_openml
import matplotlib.cm as cm
import scipy.io

# mnist = fetch_mldata('MNIST original', data_home='data/mnist')
# mnist = fetch_openml('mnist_784', version = 1, data_home='data/mnist',
#                      as_frame = False)
# 下载至https://raw.githubusercontent.com/amplab/datascience-sp14/master/lab7/mldata/mnist-original.mat
# 参考https://www.kaggle.com/avnishnish/mnist-original

mnist = scipy.io.loadmat('mnist-original.mat')

counter = 1
for i in range(1, 4):
    for j in range(1, 6):
        plt.subplot(3, 5, counter)
        plt.imshow((mnist['data'].T)[(i - 1)*8000 + j].reshape((28, 28)),
                   cmap = cm.Greys_r)
        plt.axis('off')
        counter += 1

plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


if __name__ == '__main__':
    X, y = mnist['data'].T, mnist['label'][0]
    X = X / 255.0 * 2 - 1
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, random_state = 11)

    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])

    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2,
                               verbose=1, scoring='accuracy')
    grid_search.fit(X_train[:10000], y_train[:10000])
    print('最好的打分： {}'.format(grid_search.best_score_))
    print('最好的参数集： ')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t{}: {}'.format(param_name, best_parameters[param_name]))

    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))
