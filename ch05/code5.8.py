import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
regressor = LinearRegression()
# cross_val_score 交叉验证，给出打分
scores = cross_val_score(regressor, X, y, cv=5)

print('平均打分为： {}'.format(scores.mean()))
print('打分分别为：'+'\n\t'+str(scores))

