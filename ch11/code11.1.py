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
