import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

X = []
y = []

# 图像从http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/index.html下载
for path, subdirs, files in os.walk('English/Img/GoodImg/Bmp/'):
    for filename in files:
        f = os.path.join(path, filename)
        target = filename[3:filename.index('-')]
        # 'L' 将图片转换为灰度图
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        img = Image.open(f).convert('L').resize((30, 30),
                                                resample=Image.LANCZOS)
        X.append(np.array(img).reshape(900,))
        y.append(target)

X = np.array(X)

