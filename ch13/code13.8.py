import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

# 载入图片，提取SURF描述符
all_instance_filenames = []
all_instance_targets = []

# 从https://www.microsoft.com/en-us/download/details.aspx?id=54765下载
for f in glob.glob('kagglecatsanddogs_3367a/PetImages/*/*.jpg'):
    target = 1 if 'Cat' in os.path.split(f)[1] else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)

surf_features = []
for f in all_instance_filenames:
    # 有 0K 的文件，跳过
    if os.path.getsize(f) < 1024:
        continue
    image = mh.imread(f, as_grey = True)  # 有些文件读入有问题，可能是下载的文件不合适
    # 前6个元素描述了位置和朝向
    surf_features.append(surf.surf(image)[:, 5:])
    
train_len = int(len(all_instance_filenames) * .60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]


