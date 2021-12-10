import pandas as pd

# 书中下载地址有误，应为
# http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.describe())
