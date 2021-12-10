from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   # 用于分割数据

df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]   # df最后一列为'quality'
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R^2： {}'.format(regressor.score(X_test, y_test)))
# 每次运行R^2值都不相同
