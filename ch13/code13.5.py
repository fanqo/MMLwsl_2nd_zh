import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = \
    [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image

# 从 https://github.com/PacktPublishing/Mastering-Machine-Learning-with-scikit-learn-Second-Edition/tree/master/chapter13 下载
original_img = np.array(Image.open('tree.jpg'), dtype=np.float64) / 255
original_dimensions = tuple(original_img.shape)

width, height, depth = original_dimensions
# 扁平化图片
image_flattened = np.reshape(original_img, (width * height, depth))

image_array_sample = shuffle(image_flattened, random_state = 0)[:1000]
estimator = KMeans(n_clusters = 64, random_state = 0)
estimator.fit(image_array_sample)

# 为每个像素分配到不同聚类
cluster_assignments = estimator.predict(image_flattened)
