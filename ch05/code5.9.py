import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
regressor = LinearRegression()
regressor.fit(X, y)

sample = df.sample(50)
X_test = sample[list(sample.columns)[:-1]]
y_sample = sample['quality'].values
y_prediction = regressor.predict(X_test)

for idx in range(0, 12):
    y_act = y_sample[idx]
    print('预测值为： {:.2f}，真实值为{}'
          .format(y_prediction[idx], y_act))

import matplotlib.pyplot as plt
plt.scatter(y_sample, y_prediction)
plt.axis([2, 9, 2, 9])
plt.xlabel('真实质量')
plt.ylabel('预测质量')
plt.show(block=False)
