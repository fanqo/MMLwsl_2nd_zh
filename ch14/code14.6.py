import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image
import warnings
warnings.simplefilter('ignore', Warning)

X = []
y = []

for dirpath, _, filenames in os.walk('orl_faces'):
    for filename in filenames:
        if filename[-3:] == 'pgm':
            img = Image.open(os.path.join(dirpath,
                                          filename)).convert('L')
            arr = np.array(img).reshape(10304).astype('float32') / 255
            X.append(arr)
            y.append(dirpath)

X = scale(X)

# 分隔数据集，并拟合PCA
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=150)   # 将维度降低为150

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print(X_train.shape)
print(X_train_reduced.shape)

classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)

print('交叉检验准确率： {}'.format(np.mean(accuracies)))
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print(classification_report(y_test, predictions))
