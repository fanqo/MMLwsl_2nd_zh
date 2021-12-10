import pandas as pd

# 书中下载地址有误，应为
# http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
df = pd.read_csv('winequality-red.csv', sep=';')
print('数据描述：')
print(df.describe())

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = \
    [ 'Microsoft YaHei','Adobe Heiti Std','SimHei']

plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('酒精含量')
plt.ylabel('质量')
plt.title('酒精含量与质量的关系')
plt.show(block=False)

print('相关系数矩阵：')
print(df.corr())

