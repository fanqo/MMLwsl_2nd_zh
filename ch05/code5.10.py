import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# load_boston 载入并返回 boston 房价数据集
# data.data, data.target, data.filename
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)


