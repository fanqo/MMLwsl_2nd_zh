from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print('数字： %s' % digits.target[0])
print(digits.images[0])
print('特征向量： \n %s' % digits.images[0].reshape(-1, 64))

plt.figure()
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)
plt.show()
